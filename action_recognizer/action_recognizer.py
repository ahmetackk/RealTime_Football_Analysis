import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from collections import deque
from threading import Thread, Lock, Event
from queue import Queue, Empty
import utils.profiler as profiler
from models.model_TAAD_baseline import X3D_TAAD_Baseline

class ActionRecognizer:
    """3D CNN-based action recognition with sync/async execution modes"""
    def __init__(self, model_path, device='cuda', clip_len=50, stride=25, video_width=None, video_height=None, mode='supervised', execution_mode='sync'):
        """
        Args:
            execution_mode: 'sync' (synchronous) or 'async' (parallel background processing)
        """
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
                print("[ActionRecognizer] WARNING: CUDA requested but not available. Falling back to CPU.")
                device = 'cpu'
        
        self.device = device
        print(f"[ActionRecognizer] Initializing Model... Device: {device} | Mode: {mode} | Execution: {execution_mode}")
        
        self.model = X3D_TAAD_Baseline()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(device)
        self.model.eval()
        
        self.frame_buffer = deque(maxlen=self.clip_len) 
        self.bbox_buffer = deque(maxlen=self.clip_len)  
        self.prev_logits_overlap = None 
        self.classes = ['background', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']
        self.frame_counter = 0
        
        # Async mode: background thread for parallel inference
        self._result_queue = Queue(maxsize=1)
        self._work_queue = Queue(maxsize=1)
        self._shutdown = False
        self._worker_thread = None
        
        if self.execution_mode == 'async':
            self._worker_thread = Thread(target=self._prediction_worker_v2, daemon=True)
            self._worker_thread.start()

    def set_mode(self, new_mode):
        if new_mode in ['supervised', 'unsupervised']:
            self.mode = new_mode
            print(f"[ActionRecognizer] Mode changed: {self.mode}")
    
    def shutdown(self):
        self._shutdown = True
        try:
            self._work_queue.put_nowait(None)
        except:
            pass
    
    def _prediction_worker_v2(self):
        last_processed_frame = -self.stride
        
        while not self._shutdown:
            try:
                try:
                    signal = self._work_queue.get(timeout=0.05)
                    if signal is None:
                        break
                except Empty:
                    pass
                
                current_frame = self.frame_counter
                if (len(self.frame_buffer) >= self.clip_len and 
                    current_frame - last_processed_frame >= self.stride):
                    
                    frames_snapshot = list(self.frame_buffer)
                    bbox_snapshot = list(self.bbox_buffer)
                    
                    active_ids = []
                    if len(bbox_snapshot) > 0:
                        current_bboxes = bbox_snapshot[-1]
                        active_ids = sorted(list(current_bboxes.keys()))
                    
                    if active_ids:
                        result = self._do_prediction(frames_snapshot, bbox_snapshot, active_ids)
                        
                        try:
                            try:
                                self._result_queue.get_nowait()
                            except Empty:
                                pass
                            self._result_queue.put_nowait(result)
                        except:
                            pass
                        
                        last_processed_frame = current_frame
                        
            except Exception as e:
                print(f"[ActionRecognizer] Worker error: {e}")

    def update(self, frame, detections):
        if self.execution_mode == 'sync':
            profiler.start('Action-Update')
        
        if self.video_width is None or self.video_height is None:
            h, w = frame.shape[:2]
            self.video_width = w
            self.video_height = h
            self.norm_factor_x = self.video_width / self.resize_w
            self.norm_factor_y = self.video_height / self.resize_h
        
        resized = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)
        frame_tensor = resized.astype(np.float32) / 255.0
        frame_tensor = (frame_tensor - self.norm_mean) / self.norm_std
        
        current_bboxes = {}
        if detections is not None:
            for det in detections:
                if len(det) >= 7:
                    t_id = int(det[6])
                    bbox = det[:4] 
                    current_bboxes[t_id] = bbox
        
        self.frame_buffer.append(frame_tensor)
        self.bbox_buffer.append(current_bboxes)
        self.frame_counter += 1
        
        if self.execution_mode == 'sync':
            profiler.stop('Action-Update')

    def predict(self):
        if self.execution_mode == 'sync':
            return self._predict_sync()
        else:
            return self._predict_async()
    
    def _predict_sync(self):
        """Synchronous prediction: process on main thread"""
        if len(self.frame_buffer) < self.clip_len: return {}
        if self.frame_counter % self.stride != 0: return {}

        profiler.start('Action-Predict (Total)')

        profiler.start('Action-PrepData')
        frames_np = np.array(self.frame_buffer) 
        frames_tensor = torch.from_numpy(frames_np).float().permute(3, 0, 1, 2).unsqueeze(0).to(self.device).half() 
        
        if len(self.bbox_buffer) > 0:
            current_bboxes = self.bbox_buffer[-1]
            active_ids = sorted(list(current_bboxes.keys()))
        else:
            active_ids = []

        if not active_ids: 
            profiler.stop('Action-PrepData')
            profiler.stop('Action-Predict (Total)')
            return {}
            
        rois_np, masks_np = self._prepare_rois(active_ids)
        rois_tensor = torch.from_numpy(rois_np).float().to(self.device) 
        masks_tensor = torch.from_numpy(masks_np).float().to(self.device)
        profiler.stop('Action-PrepData')

        profiler.start('Action-Inference')
        embeddings = None
        with torch.no_grad():
             with torch.autocast(device_type=self.device.split(':')[0], dtype=torch.float16):
                
                if self.mode == 'unsupervised':
                    raw_preds, embeddings = self.model([frames_tensor, rois_tensor, masks_tensor], return_embedding=True)
                else:
                    raw_preds = self.model([frames_tensor, rois_tensor, masks_tensor])
        
        current_logits = raw_preds.squeeze(0).float().cpu().numpy()
        profiler.stop('Action-Inference')
        
        profiler.start('Action-PostProcess')
        # Temporal smoothing: average overlapping logits from previous clip
        final_events = {}
        if self.prev_logits_overlap is not None:
            overlap_prev = self.prev_logits_overlap 
            overlap_curr = current_logits[:, :, :25]
            
            if overlap_prev.shape == overlap_curr.shape:
                averaged_logits = (overlap_prev + overlap_curr) / 2.0
            else:
                averaged_logits = overlap_curr 

            final_events = self._process_logits_to_events(averaged_logits, active_ids, embeddings)
        
        self.prev_logits_overlap = current_logits[:, :, 25:]
        profiler.stop('Action-PostProcess')
        
        profiler.stop('Action-Predict (Total)')
        return final_events
    
    def _prepare_rois(self, active_ids):
        """Prepare ROI tensors for each player across temporal clip (B, M, T, 5)"""
        B = 1 
        T = self.clip_len
        M = len(active_ids)
        rois_np = np.zeros((B, M, T, 5), dtype=np.float32)
        masks_np = np.zeros((B, M, T), dtype=np.float32)
        id_to_idx = {t_id: i for i, t_id in enumerate(active_ids)}
        coeff = 1.125  # Bbox expansion coefficient 
        
        vid_w = self.video_width if self.video_width else 1920
        vid_h = self.video_height if self.video_height else 1080

        if self.norm_factor_x is None: self.norm_factor_x = vid_w / self.resize_w
        if self.norm_factor_y is None: self.norm_factor_y = vid_h / self.resize_h
        
        for t_idx in range(T):
            if t_idx >= len(self.bbox_buffer): break
            frame_bboxes = self.bbox_buffer[t_idx]
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
                    
                    norm_x1 = int(tlx / self.norm_factor_x)
                    norm_y1 = int(tly / self.norm_factor_y)
                    norm_x2 = int(brx / self.norm_factor_x)
                    norm_y2 = int(bry / self.norm_factor_y)
                    
                    rois_np[0, p_idx, t_idx] = [float(t_idx), norm_x1, norm_y1, norm_x2, norm_y2]
                    masks_np[0, p_idx, t_idx] = 1.0
        return rois_np, masks_np

    def _predict_async(self):
        try:
            result = self._result_queue.get_nowait()
            return result
        except Empty:
            return {}
    
    def _do_prediction(self, frames_snapshot, bbox_snapshot, active_ids):
        try:
            profiler.start('Action-Inference (BG)')
            
            frames_np = np.array(frames_snapshot)
            frames_tensor = torch.from_numpy(frames_np).float().permute(3, 0, 1, 2).unsqueeze(0).to(self.device).half()
            
            rois_np, masks_np = self._prepare_rois_from_snapshot(bbox_snapshot, active_ids)
            rois_tensor = torch.from_numpy(rois_np).float().to(self.device)
            masks_tensor = torch.from_numpy(masks_np).float().to(self.device)
            
            embeddings = None
            with torch.no_grad():
                with torch.autocast(device_type=self.device.split(':')[0], dtype=torch.float16):
                    if self.mode == 'unsupervised':
                        raw_preds, embeddings = self.model([frames_tensor, rois_tensor, masks_tensor], return_embedding=True)
                    else:
                        raw_preds = self.model([frames_tensor, rois_tensor, masks_tensor])
            
            current_logits = raw_preds.squeeze(0).float().cpu().numpy()
            
            final_events = {}
            if self.prev_logits_overlap is not None:
                overlap_prev = self.prev_logits_overlap
                overlap_curr = current_logits[:, :, :25]
                
                if overlap_prev.shape == overlap_curr.shape:
                    averaged_logits = (overlap_prev + overlap_curr) / 2.0
                else:
                    averaged_logits = overlap_curr
                
                final_events = self._process_logits_to_events(averaged_logits, active_ids, embeddings)
            
            self.prev_logits_overlap = current_logits[:, :, 25:]
            
            profiler.stop('Action-Inference (BG)')
            return final_events
            
        except Exception as e:
            profiler.stop('Action-Inference (BG)')
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
        """Convert logits to action events using peak detection and confidence thresholding"""
        num_classes, num_players, time_steps = logits.shape
        events = {} 
        CONF_THRESH = 0.15 
        # Softmax normalization
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
                except: pass
            
            for t in range(1, time_steps - 1):
                cls = class_preds[t]
                score = scores[t]
                
                if cls == 0: continue
                if score < CONF_THRESH: continue
                
                # Peak detection: action must be local maximum
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
