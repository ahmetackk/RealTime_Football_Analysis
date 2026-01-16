import argparse
import os
import cv2
import time
from tqdm import tqdm
import supervision as sv
import torch
import numpy as np  # ✅ For embedding operations

from sports.configs.soccer import SoccerPitchConfiguration
from team_assigner.team_assigner import TeamAssigner
from visualizer.visualizer import Visualizer
from analyzer.analyzer import FootballAnalyzer
from recorder.recorder import ScenarioRecorder
# from action_recognizer.action_recognizer import ActionRecognizer
from action_recognizer.action_recognizer_optimized import ActionRecognizer
import utils.profiler as profiler
from visualizer.sidebar import SidebarVisualizer
from jersey_recognizer.jersey_recognizer import JerseyNumberRecognizer
from analyzer.discovery_engine import DiscoveryEngine
from visualizer.video_display import create_display

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    'player': os.path.join(PARENT_DIR, 'models/player_detection.engine'),
    'pitch': os.path.join(PARENT_DIR, 'models/pitch_detection.engine'),
    'ball': os.path.join(PARENT_DIR, 'models/ball_detection.engine'),
    'action': os.path.join(PARENT_DIR, 'models/action_recognition.pt') 
}

# --- RAM VIDEO LOADER ---
class RAMVideoLoader:
    def __init__(self, source):
        self.frames = []
        self.idx = 0
        print(f"\n--- VIDEO LOADING TO RAM ---")
        print(f"Source: {source}")
        
        cap = cv2.VideoCapture(source)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        pbar = tqdm(total=total, unit="frame", desc="Pre-loading")
        while True:
            ret, frame = cap.read()
            if not ret: break
            self.frames.append(frame)
            pbar.update(1)
        cap.release()
        pbar.close()
        
        self.total_frames = len(self.frames)
        print(f"--- LOADING COMPLETE ({len(self.frames)} frames) ---\n")

    def read(self):
        if self.idx < len(self.frames):
            frame = self.frames[self.idx]
            self.idx += 1
            return True, frame
        return False, None

    def release(self):
        self.frames.clear()
    
    def isOpened(self):
        return len(self.frames) > 0
# -----------------------------------------------

def main(source, target, device, mode, debug, inference_size, progress_callback=None, action_mode="supervised", execution_mode="async", display_backend="opencv"):
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, using CPU.")
            device = 'cpu'
    
    profiler.reset()
    profiler.setup(enabled=debug)

    config = SoccerPitchConfiguration()
    analyzer = FootballAnalyzer(MODELS, config, device, inference_size=inference_size)
    visualizer = Visualizer(config, display_mode=mode)
    team_assigner = TeamAssigner()
    jersey_model = JerseyNumberRecognizer(device=device)

    # Video metadata
    video_info = sv.VideoInfo.from_video_path(source)
    
    sidebar_vis = None
    if SidebarVisualizer:
        sidebar_vis = SidebarVisualizer(base_width=video_info.width, base_height=video_info.height, config=config, display_mode=mode)
        
    discovery_engine = None
    discovery_cache = {}  # ✅ Cache for discovery engine predictions
    if os.path.exists("discovery_state.pkl"):
        try:
            discovery_engine = DiscoveryEngine("discovery_state.pkl")
            print("Discovery Engine Active.")
        except: pass

    action_recognizer = None
    if ActionRecognizer and os.path.exists(MODELS['action']):
        try:
            action_recognizer = ActionRecognizer(MODELS['action'], device, 50, 25, video_info.width, video_info.height, mode=action_mode, execution_mode=execution_mode)
        except: pass

    recorder = ScenarioRecorder(config=config, fps=video_info.fps, padding=50, draw_scale=0.1)
    
    # --- RAM LOADER ---
    cap = RAMVideoLoader(source)
    total_frames = cap.total_frames
    # ------------------
    
    out_writer = None
    video_display = None
    pbar = None
    
    if mode == 'save':
        target_width = sidebar_vis.total_width if sidebar_vis else video_info.width
        target_height = sidebar_vis.height if sidebar_vis else video_info.height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(target, fourcc, video_info.fps, (target_width, target_height))
        pbar = tqdm(total=total_frames, unit="frames")
    else:
        prefer_pyqt = (display_backend == "pyqt5")
        video_display = create_display("Analysis", prefer_pyqt=prefer_pyqt)
        if sidebar_vis: video_display.setup(sidebar_vis.total_width, sidebar_vis.height)
        else: video_display.setup(video_info.width, video_info.height)

    MAX_CALIBRATION_FRAMES = 25  # Reduced from 60 - faster calibration
    total_frames_processed = 0 

    print(f"Analysis Starting... ({device})")

    try:
        target_fps = video_info.fps if video_info.fps > 0 else 25.0
        frame_duration = 1.0 / target_fps
        
        i = 0
        while True:
            loop_start = time.time()
            
            # Instant read from RAM (no disk latency)
            ret, frame = cap.read()
            if not ret:
                print("Video complete.")
                break
            
            if pbar: pbar.update(1)
            if progress_callback: progress_callback(i + 1, total_frames)
            profiler.start('main_loop') 

            if i % 2 == 0: pitch_kp = analyzer.update_pitch(frame)
            else: pitch_kp = analyzer.last_pitch_kp
            
            ball_dets = analyzer.update_ball(frame)
            player_dets = analyzer.process_players(frame, team_assigner, jersey_model)

            # ✅ ACTION RECOGNITION - OPTIMIZED (No double stride, no duplicates)
            if action_recognizer and player_dets is not None and analyzer.trained_flags:
                raw_detections = []
                if player_dets.tracker_id is not None:
                    for xyxy, conf, cls_id, t_id in zip(player_dets.xyxy, player_dets.confidence, player_dets.class_id, player_dets.tracker_id):
                        if cls_id in [0, 1]: raw_detections.append([*xyxy, conf, cls_id, t_id])
                
                # ✅ Update buffer every frame (ActionRecognizer handles stride internally)
                action_recognizer.update(frame, raw_detections)
                
                # ✅ Get predictions (will return {} if not ready)
                action_result = action_recognizer.predict()
                
                # ✅ ONLY process if this is a NEW prediction (prevent duplicates)
                if action_result and 'predictions' in action_result:
                    result_frame = action_result.get('frame', -1)
                    
                    # Check if this is a new prediction we haven't processed yet
                    if not hasattr(action_recognizer, '_last_processed_frame'):
                        action_recognizer._last_processed_frame = -1
                    
                    if result_frame > action_recognizer._last_processed_frame:
                        action_recognizer._last_processed_frame = result_frame
                        actions = action_result['predictions']
                        
                        for t_id, act_data in actions.items():
                            if t_id not in analyzer.player_stats: continue
                            
                            final_action = act_data['action']
                            final_score = act_data['score']
                            stats = analyzer.player_stats[t_id]
                            
                            TRUSTED = ['drive', 'pass', 'shot']
                            if final_action.lower() not in TRUSTED and discovery_engine:
                                try:
                                    emb = act_data.get('embedding')
                                    if emb is not None:
                                        # ✅ Create cache key from embedding hash
                                        # Ensure emb is 1D array first
                                        try:
                                            emb_flat = np.asarray(emb).flatten()
                                            # Use first 8 values as fingerprint (faster than full array hash)
                                            emb_key = tuple(np.round(emb_flat[:8], 3)) if len(emb_flat) >= 8 else tuple(emb_flat)
                                        except Exception as e:
                                            # Fallback: use hash
                                            emb_key = hash(str(emb))
                                        
                                        # ✅ Check cache first (avoid ML inference)
                                        if emb_key in discovery_cache:
                                            dname = discovery_cache[emb_key]
                                        else:
                                            # Cache miss - run discovery engine
                                            _, dname = discovery_engine.predict(emb)
                                            discovery_cache[emb_key] = dname
                                            
                                            # ✅ Limit cache size (prevent memory leak)
                                            if len(discovery_cache) > 1000:
                                                # Remove oldest entries (simple FIFO)
                                                discovery_cache.pop(next(iter(discovery_cache)))
                                        
                                        if dname.lower() != 'background':
                                            final_action = f"{dname.lower()} (DDE)"
                                except: pass
                            
                            team_id = stats.get('final_team', 0)
                            p_info = f"#{stats.get('final_number', '?')}" if stats.get('final_number') else f"ID {t_id}"
                            
                            if sidebar_vis: sidebar_vis.add_event(final_action, team_id, p_info, final_score)
                            if t_id in analyzer.player_stats: analyzer.player_stats[t_id]['current_action'] = final_action
                            
                            act_low = final_action.lower()
                            if 'background' not in act_low and final_score > 0.5:
                                act_type = 'default'
                                for k in ['pass','shot','goal','save','foul','tackle']:
                                    if k in act_low: act_type = k
                                conf = int(final_score * 100)
                                recorder.add_action(i, act_type, f"Team {team_id} {p_info}: {final_action} ({conf}%)", duration=2.0)

            # Calibration (Error-protected)
            if not analyzer.trained_flags and i >= MAX_CALIBRATION_FRAMES:
                print(f"\n[Frame {i}] Attempting calibration...")
                try:
                    if len(analyzer.calibration_crops) > 5:
                        team_assigner.fit_team_colors(analyzer.calibration_crops)
                        analyzer.trained_flags = True
                        analyzer.calibration_crops = []
                        visualizer.update_team_colors(team_assigner.team_colors[1], team_assigner.team_colors[2])
                        if sidebar_vis: sidebar_vis.update_team_colors(team_assigner.team_colors[1], team_assigner.team_colors[2])
                        print("Calibration SUCCESS.")
                    else: print("Insufficient data, postponed.")
                except Exception as e:
                    print(f"Calibration error (continuing): {e}")
                    analyzer.trained_flags = True

            profiler.start('Visualization')

            if analyzer.view_transformer: 
                recorder.add_frame(i, player_dets, ball_dets, analyzer.player_stats, analyzer.view_transformer)

            annotated_frame = visualizer.annotate_frame(frame, player_dets, ball_dets, analyzer.player_stats, pitch_kp)
            if sidebar_vis: final_frame = sidebar_vis.draw_with_radar(annotated_frame, player_dets, ball_dets, analyzer.view_transformer, visualizer.colors)
            else: final_frame = annotated_frame

            profiler.stop('Visualization')

            profiler.start('Video-Write/Show')
            if mode == 'save': out_writer.write(final_frame)
            else:
                if not video_display.show_frame(final_frame): break
            profiler.stop('Video-Write/Show')
            
            total_frames_processed += 1
            i += 1
            profiler.stop('main_loop')

            # FPS stabilizer (except save mode)
            if mode != 'save':
                elapsed = time.time() - loop_start
                wait = frame_duration - elapsed
                if wait > 0: time.sleep(wait)

    finally:
        cap.release()
        if pbar: pbar.close()
        if action_recognizer: action_recognizer.shutdown()
        if mode == 'save' and out_writer: out_writer.release()
        
        base_name = os.path.splitext(os.path.basename(source))[0]
        if len(recorder.scenes) > 0:
            recorder.save_to_file(os.path.join("results", f"{base_name}_scenario.tcb"), analyzer.player_stats)
        
        # ✅ Discovery engine cache stats
        if discovery_engine and discovery_cache:
            print(f"\n[Discovery Engine] Cache entries: {len(discovery_cache)}")
        
        print("\nREPORTING...")
        print(profiler.get_report(total_frames_processed))
        if debug: profiler.save_report(os.path.join("results", f"{base_name}_profile_log.txt"), total_frames_processed)
        
        # ✅ Proper cleanup for video display
        if video_display:
            video_display.close()
            # ✅ Extra cleanup for PyQt5
            if display_backend == 'pyqt5':
                try:
                    from PyQt5.QtWidgets import QApplication
                    app = QApplication.instance()
                    if app:
                        app.processEvents()  # Process pending events
                        app.quit()  # Quit application
                        time.sleep(0.1)  # Give time to cleanup
                except: pass
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_video_path', type=str, default=None)
    parser.add_argument('--target_video_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='stream')
    parser.add_argument('--debug', action='store_true', default=True)  # ✅ Default: True
    parser.add_argument('--inference_size', type=int, default=960)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--action_mode', type=str, default='supervised')  # ✅ Default: supervised
    parser.add_argument('--execution_mode', type=str, default='async')  # ✅ Default: async
    parser.add_argument('--display_backend', type=str, default='pyqt5')  # ✅ Default: pyqt5
    args = parser.parse_args()
    
    if args.gui or args.source_video_path is None:
        try:
            from gui.football_analysis_gui import launch_gui
            launch_gui()
        except:
            print("Usage: python main.py --source_video_path <video_path>")
    else:
        main(args.source_video_path, args.target_video_path, args.device, args.mode, args.debug, args.inference_size, action_mode=args.action_mode, execution_mode=args.execution_mode, display_backend=args.display_backend)