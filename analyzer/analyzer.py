import cv2
import time
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import Counter
from sports.common.ball import BallTracker
from sports.common.view import ViewTransformer
import utils.profiler as profiler

class FootballAnalyzer:
    """Main analyzer for player, ball, and pitch detection with team/jersey assignment"""
    def __init__(self, model_paths, config, device='cuda', inference_size=960):
        # CUDA availability check with fallback to CPU
        if device == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    print("WARNING: CUDA requested but not available. Falling back to CPU.")
                    device = 'cpu'
            except ImportError:
                print("WARNING: PyTorch not available. Using CPU.")
                device = 'cpu'
        
        self.device = 0 if device == 'cuda' else 'cpu'
        self.config = config
        self.inference_size = inference_size
        
        self.is_engine = {
            'player': model_paths['player'].endswith('.engine'),
            'pitch': model_paths['pitch'].endswith('.engine'),
            'ball': model_paths['ball'].endswith('.engine')
        }
        
        print(f"Loading models ({device})...")
        
        def load_model(path, task='detect'):
            """Load YOLO model and disable automatic saving to prevent runs/ folder creation"""
            model = YOLO(path, task=task)
            model.overrides['save'] = False
            model.overrides['save_txt'] = False
            model.overrides['save_conf'] = False
            if path.endswith('.pt'):
                model = model.to(device)
            return model
        
        self.player_model = load_model(model_paths['player'], task='detect')
        self.pitch_model = load_model(model_paths['pitch'], task='pose')
        self.ball_model = load_model(model_paths['ball'], task='detect')
        
        self.player_tracker = sv.ByteTrack(minimum_consecutive_frames=3)
        self.ball_tracker = BallTracker(buffer_size=20)
        
        self.player_stats = {}
        self.calibration_crops = []
        self.trained_flags = False
        self.view_transformer = None
        self.last_pitch_kp = None
        self.role_votes = {}
        self.frame_counter = 0

    def update_pitch(self, frame):
        """Detect pitch keypoints and create view transformer for top-down projection"""
        profiler.start('Pitch Detection')
        
        profiler.start('Pitch-Inference')
        if self.is_engine['pitch']:
            # ✅ ADD imgsz for TensorRT engine (critical for performance!)
            result = self.pitch_model.predict(
                frame, 
                verbose=False, 
                conf=0.3, 
                device=self.device, 
                save=False, 
                half=True,
                imgsz=960  # TensorRT engines need explicit size
            )[0]
        else:
            result = self.pitch_model(frame, verbose=False, conf=0.3, imgsz=self.inference_size, device=self.device, save=False)[0]
        profiler.stop('Pitch-Inference')
        
        profiler.start('Pitch-PostProcess')
        
        if result.keypoints is not None and result.keypoints.xy is not None:
            kp = sv.KeyPoints.from_ultralytics(result)
            
            if kp.xy.shape[1] > 0:
                filter_mask = kp.confidence[0] > 0.5
                valid_source = kp.xy[0][filter_mask]
                if len(valid_source) >= 4:
                    # Create perspective transform using detected keypoints
                    valid_target = np.array(self.config.vertices)[filter_mask]
                    self.view_transformer = ViewTransformer(source=valid_source, target=valid_target)
                    self.last_pitch_kp = sv.KeyPoints(xy=valid_source[np.newaxis, ...])
        
        profiler.stop('Pitch-PostProcess')
        profiler.stop('Pitch Detection')
        
        return self.last_pitch_kp

    def update_ball(self, frame):
        profiler.start('Ball Detection')
        
        profiler.start('Ball-Inference')
        if self.is_engine['ball']:
            # ✅ ADD imgsz for TensorRT engine
            result = self.ball_model.predict(
                frame, 
                conf=0.15, 
                iou=0.5, 
                verbose=False, 
                classes=[0], 
                device=self.device, 
                save=False, 
                half=True,
                imgsz=1280  # TensorRT engines need explicit size
            )[0]
        else:
            result = self.ball_model(frame, imgsz=self.inference_size, conf=0.15, iou=0.5, verbose=False, classes=[0], device=self.device, save=False)[0]
        profiler.stop('Ball-Inference')
        
        detections = sv.Detections.from_ultralytics(result)
        val = self.ball_tracker.update(detections)
        
        profiler.stop('Ball Detection')
        return val

    def process_players(self, frame, team_assigner, jersey_model):
        self.frame_counter += 1
        profiler.start('Player Logic (Total)')
        
        profiler.start('Player-Inference (YOLO)')
        
        if self.is_engine['player']:
            # ✅ ADD imgsz for TensorRT engine
            result = self.player_model.predict(
                source=frame, 
                classes=[1, 2, 3], 
                conf=0.25, 
                verbose=False,
                device=self.device,
                save=False,
                half=True,  # FP16 for speed
                imgsz=960  # TensorRT engines need explicit size
            )[0]
        else:
            result = self.player_model(
                frame, 
                imgsz=self.inference_size, 
                classes=[1, 2, 3], 
                conf=0.25, 
                verbose=False,
                device=self.device,
                save=False
            )[0]
        
        profiler.stop('Player-Inference (YOLO)')
        
        detections = sv.Detections.from_ultralytics(result)
        
        profiler.start('Player-Tracking')
        detections = self.player_tracker.update_with_detections(detections)
        profiler.stop('Player-Tracking')
        
        if detections.tracker_id is not None:
            for idx, tracker_id in enumerate(detections.tracker_id):
                t_id = int(tracker_id)
                class_id = int(detections.class_id[idx])
                
                if t_id not in self.role_votes:
                    self.role_votes[t_id] = [] 
                
                self.role_votes[t_id].append(class_id)
                self.role_votes[t_id] = self.role_votes[t_id][-50:]
        
        players_mask = (detections.class_id == 2)
        
        # Calibration phase: collect player crops for team color training
        if not self.trained_flags:
            # ✅ Reduced from 500 to 100 for faster calibration (25 frames enough)
            if self.frame_counter % 3 == 0 and len(self.calibration_crops) < 100:
                for xyxy in detections.xyxy[players_mask]:
                    x1, y1, x2, y2 = map(int, xyxy)
                    crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    if crop.size > 0: self.calibration_crops.append(crop)
            
            new_class_ids = detections.class_id.copy()
            new_class_ids[detections.class_id == 1] = 3
            new_class_ids[detections.class_id == 3] = 2
            new_class_ids[players_mask] = 0
            detections.class_id = new_class_ids
            
            profiler.stop('Player Logic (Total)')
            return detections 

        assigned_teams = []
        taken_numbers = {1: [], 2: []}
        
        if jersey_model and self.frame_counter % 10 == 0:
             for s in self.player_stats.values():
                if s['team_locked'] and s['jersey_locked']:
                    taken_numbers[s['final_team']].append(s['final_number'])

        player_indices = np.where(players_mask)[0]
        
        misdetected_gk_ref_indices = []
        
        profiler.start('Player-Loops')
        for idx in player_indices:
            tracker_id = int(detections.tracker_id[idx])
            
            # Filter out goalkeepers and referees (detected as players by YOLO)
            if self._is_goalkeeper_or_referee(tracker_id):
                assigned_teams.append(0)
                misdetected_gk_ref_indices.append(idx)
                
                if tracker_id in self.player_stats:
                    del self.player_stats[tracker_id]
                continue
            
            if tracker_id not in self.player_stats:
                self.player_stats[tracker_id] = {
                    'jersey_votes': [], 'final_number': None, 'jersey_locked': False,
                    'team_votes': [], 'final_team': None, 'team_locked': False,
                    'last_team_id': 1,
                    'strike_count': 0,
                    'jersey_fail_count': 0,
                    'last_jersey_check': 0
                }
            stats = self.player_stats[tracker_id]

            should_crop = False
            
            force_check = stats['team_locked'] and (self.frame_counter % 15 == 0)
            check_team = (not stats['team_locked']) and (self.frame_counter % 4 == 0)
            
            # Adaptive jersey checking: increase interval based on failure count
            check_jersey = False
            if jersey_model and not stats['jersey_locked'] and stats['team_locked']:
                fail_count = stats.get('jersey_fail_count', 0)
                
                if fail_count < 5:
                    jersey_interval = 15
                elif fail_count < 10:
                    jersey_interval = 30
                elif fail_count < 20:
                    jersey_interval = 60
                else:
                    jersey_interval = 90
                
                frames_since_last = self.frame_counter - stats.get('last_jersey_check', 0)
                if frames_since_last >= jersey_interval:
                    check_jersey = True
            
            if force_check or check_team or check_jersey:
                should_crop = True

            crop = None
            if should_crop:
                profiler.start('Player-Crop')
                crop = self._get_crop(frame, detections.xyxy[idx])
                profiler.stop('Player-Crop')

            profiler.start('Player-TeamLogic')
            
            if crop is not None and crop.size > 0:
                current_team = self._process_team_logic(stats, crop, team_assigner)
                stats['last_team_id'] = current_team
                assigned_teams.append(current_team - 1)
            
            elif stats['team_locked']:
                assigned_teams.append(stats['final_team'] - 1)
                
            else:
                assigned_teams.append(stats['last_team_id'] - 1)
                
            profiler.stop('Player-TeamLogic')

            profiler.start('Player-JerseyLogic')
            if check_jersey and crop is not None and stats['team_locked']:
                stats['last_jersey_check'] = self.frame_counter
                result = self._process_jersey_logic(stats, crop, jersey_model, taken_numbers, team_assigner)
                
                if result == 1:
                    stats['jersey_fail_count'] = 0
                elif result == -1:
                    stats['jersey_fail_count'] = stats.get('jersey_fail_count', 0) + 1
            profiler.stop('Player-JerseyLogic')

        profiler.stop('Player-Loops')

        new_class_ids = detections.class_id.copy()
        new_class_ids[players_mask] = np.array(assigned_teams)
        new_class_ids[detections.class_id == 3] = 2
        new_class_ids[detections.class_id == 1] = 3

        for idx in misdetected_gk_ref_indices:
            tracker_id = int(detections.tracker_id[idx])
            votes = self.role_votes.get(tracker_id, [])
            gk_votes = votes.count(1) 
            ref_votes = votes.count(3)
            
            if gk_votes >= ref_votes:
                new_class_ids[idx] = 3
            else:
                new_class_ids[idx] = 2
        
        detections.class_id = new_class_ids
        
        profiler.stop('Player Logic (Total)')
        return detections

    def _get_crop(self, frame, xyxy):
        x1, y1, x2, y2 = map(int, xyxy)
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]

    def _process_team_logic(self, stats, crop, team_assigner):
        """Team assignment with voting system and strike-based unlocking"""
        vote = team_assigner.get_player_team(crop)
        
        # If team is locked, check for consistency (strike system)
        if stats['team_locked']:
            if vote != stats['final_team']:
                stats['strike_count'] = stats.get('strike_count', 0) + 1
            else:
                stats['strike_count'] = max(0, stats.get('strike_count', 0) - 1)
            
            if stats['strike_count'] >= 5:
                stats['team_locked'] = False
                stats['final_team'] = None
                stats['team_votes'] = []
                stats['strike_count'] = 0
                stats['jersey_locked'] = False
                stats['final_number'] = None
                return vote
            
            return stats['final_team']

        # Voting system: lock team when 70% consensus reached
        stats['team_votes'].append(vote)
        votes = stats['team_votes'][-15:]
        
        if len(votes) >= 8:
            most_common = Counter(votes).most_common(1)[0]
            vote_ratio = most_common[1] / len(votes)
            
            if vote_ratio >= 0.70:
                stats['final_team'] = most_common[0]
                stats['team_locked'] = True
                stats['strike_count'] = 0
                return most_common[0]
        
        return vote

    def _is_goalkeeper_or_referee(self, tracker_id):
        if tracker_id not in self.role_votes:
            return False
        
        votes = self.role_votes[tracker_id]
        
        if len(votes) < 10:
            return False
        
        goalkeeper_votes = votes.count(1)
        player_votes = votes.count(2)
        referee_votes = votes.count(3)
        
        total_votes = len(votes)
        non_player_votes = goalkeeper_votes + referee_votes
        non_player_ratio = non_player_votes / total_votes

        if non_player_ratio >= 0.60:
            return True
        
        return False

    def _process_jersey_logic(self, stats, crop, jersey_model, taken_numbers, team_assigner):
        """Jersey number recognition with voting and duplicate prevention"""
        h_crop, w_crop = crop.shape[:2]
        
        if h_crop < 50 or w_crop < 25:
            return 0
        
        # Upscale small crops for better recognition
        if h_crop < 100:
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        try:
            res = jersey_model.predict(crop)
            
            if not res['is_legible']:
                return -1
            
            if res['confidence'] < 0.6:
                return 0
            
            num = res['number']
            
            if num and num.isdigit():
                num_int = int(num)
                if 1 <= num_int <= 99:
                    num = str(num_int)
                    
                    # Check for duplicate numbers within same team
                    if num not in taken_numbers[stats['final_team']]:
                        stats['jersey_votes'].append(num)
                        
                        # Lock jersey number when 3+ votes out of 5
                        if len(stats['jersey_votes']) >= 5:
                            common = Counter(stats['jersey_votes']).most_common(1)[0]
                            if common[1] >= 3:
                                stats['final_number'] = common[0]
                                stats['jersey_locked'] = True
                                taken_numbers[stats['final_team']].append(common[0])
                        return 1
                    else:
                        stats['jersey_votes'] = []
                        return 0
            
            return 0
        except:
            return -1
    
    def _enhance_number_contrast(self, crop, team_id, team_assigner):
        team_color = team_assigner.team_colors.get(team_id, np.array([128, 128, 128]))
        team_brightness = np.mean(team_color)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        is_dark_jersey = team_brightness < 128
        
        if is_dark_jersey:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced_gray = clahe.apply(gray)
        else:
            inverted = 255 - gray
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced_inv = clahe.apply(inverted)
            enhanced_gray = 255 - enhanced_inv
        
        enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        
        return enhanced