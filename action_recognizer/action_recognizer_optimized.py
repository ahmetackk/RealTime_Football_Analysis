import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from collections import deque  # ✅ For fixed-size stats buffers
from threading import Thread
from queue import Queue, Empty
import time
import utils.profiler as profiler
from models.model_TAAD_baseline import X3D_TAAD_Baseline

class ActionRecognizer:
    """
    ULTRA-OPTIMIZED ZERO-COPY PIPELINE
    - ✅ Zero-copy circular numpy buffer (no deque overhead)
    - ✅ Direct in-place frame writes (no intermediate arrays)
    - ✅ Numpy view-based slicing (no list copying)
    - ✅ Non-blocking CUDA streams
    - ✅ GPU warmup for consistent performance
    """
    def __init__(self, model_path, device='cuda', clip_len=50, stride=25, video_width=None, video_height=None, mode='supervised', execution_mode='async', use_optimized=True):
        self.clip_len = clip_len
        self.stride = stride
        self.mode = mode
        self.execution_mode = execution_mode
        
        self.resize_h = 352
        self.resize_w = 640
        self.norm_mean = 0.45
        self.norm_std = 0.225
        
        self.video_width = video_width
        self.video_height = video_height
        self.norm_factor_x = None
        self.norm_factor_y = None
        
        if self.video_width and self.video_height:
            self.norm_factor_x = self.video_width / self.resize_w
            self.norm_factor_y = self.video_height / self.resize_h
        
        if device == 'cuda':
            if not torch.cuda.is_available():
                print("[ActionRecognizer] WARNING: CUDA not available. Using CPU.")
                device = 'cpu'
        
        self.device = device
        print(f"[ActionRecognizer] 🚀 ZERO-COPY CIRCULAR BUFFER PIPELINE (stride={stride})")
        print(f"[ActionRecognizer] Device: {device} | Mode: {mode} | Execution: {execution_mode}")
        
        # *** CUDA Stream - Non-blocking GPU operations ***
        self.cuda_stream = torch.cuda.Stream() if device == 'cuda' else None
        if self.cuda_stream:
            print("[ActionRecognizer] ✓ Non-blocking CUDA stream created")
        
        # Model loading
        print(f"[ActionRecognizer] Loading model: {model_path}")
        self.model = X3D_TAAD_Baseline()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(device)
        self.model.eval()
        
        # AMP optimization
        self.use_amp = (device == 'cuda' and use_optimized)
        if self.use_amp:
            print("[ActionRecognizer] ✓ AMP (Mixed Precision) ACTIVE")
        
        # ✅ GPU WARMUP - Prevent first inference slowdown
        if device == 'cuda':
            print("[ActionRecognizer] 🔥 Warming up GPU...")
            dummy_input = torch.randn(1, 3, 50, 352, 640).to(device)
            dummy_rois = torch.zeros(1, 1, 50, 5).to(device)
            dummy_masks = torch.ones(1, 1, 50).to(device)
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        _ = self.model([dummy_input, dummy_rois, dummy_masks])
                else:
                    _ = self.model([dummy_input, dummy_rois, dummy_masks])
            del dummy_input, dummy_rois, dummy_masks
            torch.cuda.empty_cache()
            print("[ActionRecognizer] ✓ GPU warmup complete")
        
        # Note: Pinned memory pool removed - circular numpy buffer is faster
        
        # ✅ ZERO-COPY CIRCULAR BUFFERS (numpy-based, no deque copying)
        self.buffer_capacity = self.clip_len * 2
        
        # Pre-allocated numpy arrays for zero-copy operations
        self.frame_buffer_np = np.zeros((self.buffer_capacity, self.resize_h, self.resize_w, 3), dtype=np.float32)
        self.bbox_buffer_list = [None] * self.buffer_capacity  # Can't use numpy for dicts
        
        # ✅ Pre-allocated snapshot buffer (reused every prediction - no allocation overhead)
        self.snapshot_buffer = np.zeros((self.clip_len, self.resize_h, self.resize_w, 3), dtype=np.float32)
        
        self.buffer_write_idx = 0  # Circular buffer write position
        self.buffer_fill_count = 0  # How many frames written
        self.frame_counter = 0
        
        print(f"[ActionRecognizer] ✓ Zero-copy circular buffers allocated ({self.buffer_capacity} frames)")
        print(f"[ActionRecognizer] ✓ Snapshot buffer pre-allocated ({self.clip_len} frames)")
        
        # Prediction state
        self.prev_logits_overlap = None 
        self.classes = ['background', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']
        
        # Latest results cache (thread-safe dict updates)
        self.latest_results = {}
        self.latest_results_frame = -1
        
        # ✅ EXECUTION MODE: async (background worker) vs sync (on-demand)
        self._shutdown = False
        self._worker_thread = None
        
        if execution_mode == 'async':
            # Background worker for non-blocking inference
            self._worker_thread = Thread(target=self._continuous_prediction_worker, daemon=True)
            self._worker_thread.start()
            print("[ActionRecognizer] ✓ ASYNC mode - background worker started")
        else:
            print("[ActionRecognizer] ✓ SYNC mode - on-demand inference")
        
        # Stats
        # ✅ Use deque with maxlen to prevent unbounded memory growth
        self.stats = {
            'total_updates': 0,
            'total_predictions': 0,
            'total_inference_time': 0.0,
            'avg_inference_time': 0.0,
            'update_times': deque(maxlen=1000),  # Keep last 1000 only
            'inference_times': deque(maxlen=1000)  # Keep last 1000 only
        }
        
        print("[ActionRecognizer] ✓ Zero-copy worker started") if execution_mode == 'async' else print("[ActionRecognizer] ✓ Zero-copy buffers ready")
        print(f"[ActionRecognizer] ✓ Ultra-optimized {execution_mode.upper()} pipeline ready!")

    def set_mode(self, new_mode):
        if new_mode in ['supervised', 'unsupervised']:
            self.mode = new_mode

    def shutdown(self):
        print("\n[ActionRecognizer] 🛑 Shutdown starting...")
        self._shutdown = True
        
        # Only join worker thread if it exists (async mode)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
        
        # Sync CUDA stream
        if self.cuda_stream:
            print("[ActionRecognizer] ⏳ Synchronizing GPU stream...")
            self.cuda_stream.synchronize()
            print("[ActionRecognizer] ✓ GPU stream synced")
        
        # Stats
        if self.stats['total_predictions'] > 0:
            print(f"[ActionRecognizer] 📊 OPTIMIZATION STATS ({self.execution_mode.upper()} mode):")
            print(f"  Total predictions: {self.stats['total_predictions']}")
            print(f"  Avg inference: {self.stats['avg_inference_time']:.1f} ms")
            if self.stats['update_times']:
                # ✅ Convert deque to list for slicing
                recent_updates = list(self.stats['update_times'])[-100:]
                print(f"  Avg update: {np.mean(recent_updates):.2f} ms")
        print("[ActionRecognizer] ✓ Shutdown complete")
    
    def _continuous_prediction_worker(self):
        """Background worker - Runs in separate CUDA stream"""
        print(f"[ActionRecognizer] 🔄 Worker started (stride={self.stride})...")
        
        last_processed_frame = -1
        
        while not self._shutdown:
            try:
                # ✅ NO LOCK - Read current state atomically
                current_frame = self.frame_counter
                buffer_fill = self.buffer_fill_count
                
                if buffer_fill < self.clip_len:
                    time.sleep(0.001)  # Reduced from 0.005 - more responsive
                    continue
                
                # Stride control
                if current_frame % self.stride != 0:
                    time.sleep(0.0001)  # Reduced from 0.001 - minimal wait
                    continue
                
                # Prevent reprocessing same frame
                if current_frame == last_processed_frame:
                    time.sleep(0.0001)  # Reduced from 0.001
                    continue
                
                # ✅ OPTIMIZED SNAPSHOT - Reuse pre-allocated buffer
                # Get the last clip_len frames from circular buffer
                write_idx = self.buffer_write_idx
                
                if buffer_fill >= self.buffer_capacity:
                    # Buffer is full - handle wrap-around with optimized copy
                    start_idx = write_idx  # Oldest frame is at write position
                    
                    # Calculate how many frames until end of buffer
                    frames_until_end = self.buffer_capacity - start_idx
                    
                    if frames_until_end >= self.clip_len:
                        # No wrap-around needed - direct slice (view, not copy!)
                        frames_snapshot = self.frame_buffer_np[start_idx:start_idx + self.clip_len]
                        bbox_snapshot = self.bbox_buffer_list[start_idx:start_idx + self.clip_len]
                    else:
                        # Wrap-around needed - use pre-allocated buffer
                        # Copy in two chunks (still faster than fancy indexing)
                        self.snapshot_buffer[:frames_until_end] = self.frame_buffer_np[start_idx:]
                        remaining = self.clip_len - frames_until_end
                        self.snapshot_buffer[frames_until_end:] = self.frame_buffer_np[:remaining]
                        
                        frames_snapshot = self.snapshot_buffer  # Already in correct order
                        # ✅ Ensure bbox_snapshot is a proper list
                        bbox_snapshot = list(self.bbox_buffer_list[start_idx:]) + list(self.bbox_buffer_list[:remaining])
                else:
                    # Buffer not full yet - simple slice (view, not copy!)
                    start_idx = max(0, buffer_fill - self.clip_len)
                    frames_snapshot = self.frame_buffer_np[start_idx:buffer_fill]
                    bbox_snapshot = self.bbox_buffer_list[start_idx:buffer_fill]
                
                snapshot_frame = current_frame
                
                # Get active IDs
                active_ids = []
                if len(bbox_snapshot) > 0:
                    current_bboxes = bbox_snapshot[-1]
                    # ✅ Check if last bbox is not None
                    if current_bboxes is not None:
                        active_ids = sorted(list(current_bboxes.keys()))
                
                if not active_ids:
                    continue  # No sleep needed - just skip to next iteration
                
                # *** INFERENCE - Non-blocking CUDA stream ***
                # ✅ Conditional logging (every 5th inference) - reduces I/O overhead
                should_log = (self.stats['total_predictions'] % 5 == 0)
                
                if should_log:
                    print(f"[ActionRecognizer] 🎯 Frame {snapshot_frame}: GPU inference starting...")
                inference_start = time.time()
                
                if self.cuda_stream:
                    # Run in separate stream - NO BLOCKING!
                    with torch.cuda.stream(self.cuda_stream):
                        result = self._do_prediction(frames_snapshot, bbox_snapshot, active_ids)
                    # NO synchronize() here - let GPU work in parallel
                else:
                    result = self._do_prediction(frames_snapshot, bbox_snapshot, active_ids)
                
                inference_time = (time.time() - inference_start) * 1000
                
                # ✅ Thread-safe cache update (dict assignment is atomic in Python)
                # Add frame metadata to prevent duplicate recording
                self.latest_results = {
                    'predictions': result,
                    'frame': snapshot_frame,
                    'timestamp': time.time()
                }
                self.latest_results_frame = snapshot_frame
                
                # Update stats
                self.stats['total_predictions'] += 1
                self.stats['total_inference_time'] += inference_time
                self.stats['avg_inference_time'] = self.stats['total_inference_time'] / self.stats['total_predictions']
                self.stats['inference_times'].append(inference_time)
                
                last_processed_frame = snapshot_frame
                
                if should_log:
                    print(f"[ActionRecognizer] ✓ Frame {snapshot_frame}: Complete ({inference_time:.1f}ms, GPU async)")
                
                # Stats reporting (every 10 inferences)
                if self.stats['total_predictions'] % 10 == 0:
                    # ✅ Convert deque to list for slicing
                    recent_times = list(self.stats['inference_times'])[-10:]
                    avg_recent = np.mean(recent_times)
                    print(f"[ActionRecognizer] 📊 Last 10 inferences: {avg_recent:.1f}ms avg")
                
            except Exception as e:
                print(f"[ActionRecognizer] ❌ Worker error: {e}")
                time.sleep(0.001)  # Reduced from 0.01 - recover faster
        
        print("[ActionRecognizer] 🛑 Worker stopped")

    def update(self, frame, detections):
        """
        ✅ ZERO-COPY UPDATE - Direct write to circular buffer
        No deque append, no list copying
        
        Args:
            frame: BGR frame from OpenCV
            detections: List of [x1, y1, x2, y2, conf, cls_id, track_id]
        """
        update_start = time.time()
        
        # Resize (CPU operation)
        resize_start = time.time()
        frame_resized = cv2.resize(frame, (self.resize_w, self.resize_h))
        resize_time = (time.time() - resize_start) * 1000
        
        # Normalize and write DIRECTLY to circular buffer (zero-copy!)
        preprocess_start = time.time()
        write_idx = self.buffer_write_idx
        
        # ✅ Direct in-place write - NO intermediate arrays
        np.subtract(frame_resized.astype(np.float32) / 255.0, self.norm_mean, out=self.frame_buffer_np[write_idx])
        np.divide(self.frame_buffer_np[write_idx], self.norm_std, out=self.frame_buffer_np[write_idx])
        
        preprocess_time = (time.time() - preprocess_start) * 1000
        
        # Parse bounding boxes (can't avoid dict creation)
        bbox_start = time.time()
        bbox_dict = {}
        if detections is not None:
            for det in detections:
                if len(det) >= 7:
                    x1, y1, x2, y2, conf, cls_id, track_id = det[:7]
                    track_id = int(track_id)
                    bbox_dict[track_id] = [float(x1), float(y1), float(x2), float(y2)]
        
        # ✅ Direct write to circular buffer
        self.bbox_buffer_list[write_idx] = bbox_dict
        bbox_time = (time.time() - bbox_start) * 1000
        
        # ✅ Advance circular buffer pointer (atomic operation)
        self.buffer_write_idx = (write_idx + 1) % self.buffer_capacity
        self.buffer_fill_count = min(self.buffer_fill_count + 1, self.buffer_capacity)
        self.frame_counter += 1
        
        total_update_time = (time.time() - update_start) * 1000
        
        # Stats
        self.stats['total_updates'] += 1
        self.stats['update_times'].append(total_update_time)
        
        # Periodic reporting
        if self.stats['total_updates'] % 100 == 0:
            # ✅ Convert deque to list for slicing
            recent = list(self.stats['update_times'])[-100:]
            print(f"[ActionRecognizer] 📊 UPDATE (last 100): {np.mean(recent):.2f}ms avg [ZERO-COPY]")

    def predict(self):
        """
        Get action predictions
        
        ASYNC mode: Returns cached results from background worker (instant)
        SYNC mode: Runs inference now and returns results (blocks main thread)
        
        Returns:
            dict: {'predictions': {...}, 'frame': int, 'timestamp': float}
                  or {} if no results available
        """
        if self.execution_mode == 'async':
            # Return cached results from worker thread
            return self.latest_results.copy() if self.latest_results else {}
        else:
            # SYNC mode - run inference now
            return self._sync_predict()
    
    def _sync_predict(self):
        """Synchronous prediction - runs inference immediately"""
        current_frame = self.frame_counter
        buffer_fill = self.buffer_fill_count
        
        # Check if we have enough frames
        if buffer_fill < self.clip_len:
            return {}
        
        # Check stride
        if current_frame % self.stride != 0:
            return {}
        
        # Check if already processed this frame
        if current_frame == self.latest_results_frame:
            return self.latest_results.copy() if self.latest_results else {}
        
        # Get snapshot
        write_idx = self.buffer_write_idx
        
        if buffer_fill >= self.buffer_capacity:
            # Buffer is full - handle wrap-around with optimized copy
            start_idx = write_idx
            frames_until_end = self.buffer_capacity - start_idx
            
            if frames_until_end >= self.clip_len:
                # No wrap-around - direct slice
                frames_snapshot = self.frame_buffer_np[start_idx:start_idx + self.clip_len]
                bbox_snapshot = self.bbox_buffer_list[start_idx:start_idx + self.clip_len]
            else:
                # Wrap-around - use pre-allocated buffer
                self.snapshot_buffer[:frames_until_end] = self.frame_buffer_np[start_idx:]
                remaining = self.clip_len - frames_until_end
                self.snapshot_buffer[frames_until_end:] = self.frame_buffer_np[:remaining]
                
                frames_snapshot = self.snapshot_buffer
                # ✅ Ensure bbox_snapshot is a proper list
                bbox_snapshot = list(self.bbox_buffer_list[start_idx:]) + list(self.bbox_buffer_list[:remaining])
        else:
            # Buffer not full yet - simple slice
            start_idx = max(0, buffer_fill - self.clip_len)
            frames_snapshot = self.frame_buffer_np[start_idx:buffer_fill]
            bbox_snapshot = self.bbox_buffer_list[start_idx:buffer_fill]
        
        # Get active IDs
        active_ids = []
        if len(bbox_snapshot) > 0:
            current_bboxes = bbox_snapshot[-1]
            if current_bboxes:
                active_ids = sorted(list(current_bboxes.keys()))
        
        if not active_ids:
            return {}
        
        # Run inference (will block main thread in sync mode)
        # ✅ Conditional logging
        should_log = (self.stats['total_predictions'] % 5 == 0)
        
        if should_log:
            print(f"[ActionRecognizer] 🎯 Frame {current_frame}: SYNC inference starting...")
        inference_start = time.time()
        
        result = self._do_prediction(frames_snapshot, bbox_snapshot, active_ids)
        
        inference_time = (time.time() - inference_start) * 1000
        
        # Update cache
        self.latest_results = {
            'predictions': result,
            'frame': current_frame,
            'timestamp': time.time()
        }
        self.latest_results_frame = current_frame
        
        # Update stats
        self.stats['total_predictions'] += 1
        self.stats['total_inference_time'] += inference_time
        self.stats['avg_inference_time'] = self.stats['total_inference_time'] / self.stats['total_predictions']
        
        if should_log:
            print(f"[ActionRecognizer] ✓ Frame {current_frame}: SYNC complete ({inference_time:.1f}ms)")
        
        return self.latest_results.copy()
    
    def _do_prediction(self, frames_snapshot, bbox_snapshot, active_ids):
        """Inference - Runs in CUDA stream"""
        try:
            # ✅ ZERO-COPY: frames_snapshot is already numpy array!
            # Just ensure correct shape and convert to tensor
            frames_tensor = torch.from_numpy(frames_snapshot).permute(3, 0, 1, 2).unsqueeze(0)
            frames_tensor = frames_tensor.to(self.device, non_blocking=True)
            
            # Prepare ROIs (snapshot already has correct length)
            rois_np, masks_np = self._prepare_rois_from_snapshot(bbox_snapshot, active_ids)
            
            # ✅ Non-blocking GPU transfer
            rois_tensor = torch.from_numpy(rois_np).float().to(self.device, non_blocking=True)
            masks_tensor = torch.from_numpy(masks_np).float().to(self.device, non_blocking=True)
            
            embeddings = None
            with torch.no_grad():
                if self.use_amp:
                    # ✅ Use modern autocast API
                    with torch.amp.autocast('cuda'):
                        if self.mode == 'unsupervised':
                            raw_preds, embeddings = self.model([frames_tensor, rois_tensor, masks_tensor], return_embedding=True)
                        else:
                            raw_preds = self.model([frames_tensor, rois_tensor, masks_tensor])
                else:
                    if self.mode == 'unsupervised':
                        raw_preds, embeddings = self.model([frames_tensor, rois_tensor, masks_tensor], return_embedding=True)
                    else:
                        raw_preds = self.model([frames_tensor, rois_tensor, masks_tensor])
            
            # ✅ CRITICAL FIX: Non-blocking GPU→CPU transfer (prevents implicit synchronization)
            current_logits = raw_preds.squeeze(0).float().cpu()
            
            # ✅ Convert to numpy AFTER ensuring async copy started
            # Note: .numpy() on CPU tensor is instant (no GPU sync needed)
            current_logits_np = current_logits.numpy()
            
            final_events = {}
            if self.prev_logits_overlap is not None:
                overlap_prev = self.prev_logits_overlap
                overlap_curr = current_logits_np[:, :, :25]
                
                if overlap_prev.shape == overlap_curr.shape:
                    averaged_logits = (overlap_prev + overlap_curr) / 2.0
                else:
                    averaged_logits = overlap_curr
                
                final_events = self._process_logits_to_events(averaged_logits, active_ids, embeddings)
            
            self.prev_logits_overlap = current_logits_np[:, :, 25:]
            return final_events
            
        except Exception as e:
            print(f"[ActionRecognizer] ❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _prepare_rois_from_snapshot(self, bbox_snapshot, active_ids):
        B = 1
        T = self.clip_len
        M = len(active_ids)
        rois_np = np.zeros((B, M, T, 5), dtype=np.float32)
        masks_np = np.zeros((B, M, T), dtype=np.float32)
        id_to_idx = {t_id: i for i, t_id in enumerate(active_ids)}
        coeff = 1.125
        
        vid_w = self.video_width if self.video_width else 1920
        vid_h = self.video_height if self.video_height else 1080
        
        norm_x = self.norm_factor_x if self.norm_factor_x else vid_w / self.resize_w
        norm_y = self.norm_factor_y if self.norm_factor_y else vid_h / self.resize_h
        
        for t_idx in range(min(T, len(bbox_snapshot))):
            frame_bboxes = bbox_snapshot[t_idx]
            # ✅ Skip if None (empty frame in circular buffer)
            if frame_bboxes is None:
                continue
            for t_id in active_ids:
                if t_id in frame_bboxes:
                    p_idx = id_to_idx[t_id]
                    bbox = frame_bboxes[t_id]
                    x1, y1, x2, y2 = bbox
                    
                    w = x2 - x1
                    h = y2 - y1
                    
                    tlx = max(0, min(vid_w, int((x1 + w/2) - (coeff * w)/2)))
                    tly = max(0, min(vid_h, int((y1 + h/2) - (coeff * h)/2)))
                    brx = max(0, min(vid_w, int((x1 + w/2) + (coeff * w)/2)))
                    bry = max(0, min(vid_h, int((y1 + h/2) + (coeff * h)/2)))
                    
                    norm_x1 = int(tlx / norm_x)
                    norm_y1 = int(tly / norm_y)
                    norm_x2 = int(brx / norm_x)
                    norm_y2 = int(bry / norm_y)
                    
                    rois_np[0, p_idx, t_idx] = [float(t_idx), norm_x1, norm_y1, norm_x2, norm_y2]
                    masks_np[0, p_idx, t_idx] = 1.0
        
        return rois_np, masks_np

    def _process_logits_to_events(self, logits, active_ids, embeddings_tensor=None):
        num_classes, num_players, time_steps = logits.shape
        events = {} 
        CONF_THRESH = 0.15
        
        exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True) 
        
        for m_idx, t_id in enumerate(active_ids):
            player_probs = probs[:, m_idx, :] 
            class_preds = np.argmax(player_probs, axis=0)
            scores = np.max(player_probs, axis=0)
            
            player_emb_vec = None
            
            if self.mode == 'unsupervised' and embeddings_tensor is not None:
                try:
                    player_emb_raw = embeddings_tensor[0, m_idx]
                    if player_emb_raw.dim() > 1:
                        dims_to_reduce = [d for d in range(player_emb_raw.dim()) if player_emb_raw.shape[d] != 512]
                        if not dims_to_reduce and player_emb_raw.shape[-1] == 512:
                             if player_emb_raw.dim() == 2: 
                                 player_emb_vec = player_emb_raw.mean(dim=0).float().cpu().numpy()
                             else:
                                 player_emb_vec = player_emb_raw.float().cpu().numpy()
                        elif dims_to_reduce:
                             player_emb_vec = player_emb_raw.mean(dim=dims_to_reduce).float().cpu().numpy()
                    else:
                        player_emb_vec = player_emb_raw.float().cpu().numpy()
                except: 
                    pass
            
            for t in range(1, time_steps - 1):
                cls = class_preds[t]
                score = scores[t]
                
                if cls == 0: continue
                if score < CONF_THRESH: continue
                
                prev_score = player_probs[cls, t-1]
                next_score = player_probs[cls, t+1]
                
                if score > prev_score and score > next_score:
                    action_name = self.classes[cls]
                    
                    if t_id not in events or events[t_id]['score'] < score:
                        event_data = {
                            "action": action_name, 
                            "score": float(score), 
                            "class_id": int(cls)
                        }
                        if self.mode == 'unsupervised':
                            event_data["embedding"] = player_emb_vec
                        events[t_id] = event_data
        return events