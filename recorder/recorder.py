import pickle
import uuid
import numpy as np
import supervision as sv

class ScenarioRecorder:
    def __init__(self, config, fps=30.0, padding=50, draw_scale=0.1):
        self.scenes = []
        self.actions = []
        self.fps = fps
        self.padding = padding
        self.scale = draw_scale
        self.pitch_width = config.width
        self.pitch_length = config.length
        self.id_map = {} 

    def _get_uuid(self, tracker_id):
        if tracker_id not in self.id_map:
            self.id_map[tracker_id] = str(uuid.uuid4())
        return self.id_map[tracker_id]

    def add_action(self, frame_idx: int, action_type: str, text: str, duration: float = 3.0):
        timestamp = float(frame_idx) / self.fps
        action = {
            'timestamp': timestamp,
            'type': action_type,
            'text': text,
            'duration': duration
        }
        self.actions.append(action)
        print(f"Action recorded: [{action_type}] {text} @ {timestamp:.2f}s")

    def _transform_and_clamp(self, xy_real):
        x_real, y_real = xy_real
        x_real = max(0, min(x_real, self.pitch_length))
        y_real = max(0, min(y_real, self.pitch_width))
        x_gui = (x_real * self.scale) + self.padding
        y_gui = (y_real * self.scale) + self.padding
        return (float(x_gui), float(y_gui))

    def add_frame(self, frame_idx, detections, ball_detections, player_stats, view_transformer):
        if view_transformer is None:
            return

        timestamp = float(frame_idx) / self.fps
        
        ball_pos = None
        if len(ball_detections) > 0:
            ball_xy_pixel = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            ball_xy_real = view_transformer.transform_points(points=ball_xy_pixel)[0]
            ball_pos = self._transform_and_clamp(ball_xy_real)
        
        players_data = []
        players_xy_pixel = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy_real = view_transformer.transform_points(points=players_xy_pixel)

        for i, tracker_id in enumerate(detections.tracker_id):
            tracker_id = int(tracker_id)
            raw_class_id = int(detections.class_id[i])

            if raw_class_id == 2:
                class_id = 3
                team_id = 9
                jersey_num = 0
            elif raw_class_id == 3:
                class_id = 1
                team_id = 0
                jersey_num = 0
                if tracker_id in player_stats:
                    stats = player_stats[tracker_id]
                    if stats.get('final_team'):
                        team_id = 0 if stats['final_team'] == 1 else 1
                    elif len(stats.get('team_votes', [])) > 0:
                        last_vote = stats['team_votes'][-1]
                        team_id = 0 if last_vote == 1 else 1
                    if stats.get('final_number') is not None:
                        jersey_num = int(stats['final_number'])
                    elif len(stats.get('jersey_votes', [])) > 0:
                        from collections import Counter
                        votes = stats['jersey_votes']
                        jersey_num = Counter(votes).most_common(1)[0][0]
            else:
                class_id = 2
                team_id = raw_class_id
                jersey_num = 0
                
                if tracker_id in player_stats:
                    stats = player_stats[tracker_id]
                    
                    if stats.get('final_team'):
                        team_id = 0 if stats['final_team'] == 1 else 1
                    elif len(stats.get('team_votes', [])) > 0:
                        last_vote = stats['team_votes'][-1]
                        team_id = 0 if last_vote == 1 else 1

                    if stats.get('final_number') is not None:
                        jersey_num = int(stats['final_number'])
                    elif len(stats.get('jersey_votes', [])) > 0:
                        from collections import Counter
                        votes = stats['jersey_votes']
                        jersey_num = Counter(votes).most_common(1)[0][0]
            
            xy_real = players_xy_real[i]
            pos_gui = self._transform_and_clamp(xy_real)

            p_data = {
                'id': self._get_uuid(tracker_id),
                'tracker_id': tracker_id,
                'team': team_id,
                'jersey': jersey_num,
                'pos': pos_gui,
                'class_id': class_id
            }
            players_data.append(p_data)

        scene = {
            'timestamp': timestamp,
            'state': {
                'ball_pos': ball_pos,
                'players': players_data
            },
            'listbox_id': f"Time: {timestamp:.2f}s"
        }
        self.scenes.append(scene)

    def save_to_file(self, filepath, final_player_stats):
        from collections import Counter
        
        print("Performing retrospective team correction...")
        
        count_updates = 0
        count_team_updates = 0
        
        for tracker_id, stats in final_player_stats.items():
            if tracker_id not in self.id_map:
                continue
                
            target_uuid = self.id_map[tracker_id]
            
            best_team = stats.get('final_team')
            team_votes = stats.get('team_votes', [])
            
            if best_team is None and len(team_votes) > 0:
                vote_counts = Counter(team_votes)
                most_common = vote_counts.most_common(1)[0]
                best_team = most_common[0]
            
            if best_team is None:
                continue

            for scene in self.scenes:
                for player in scene['state']['players']:
                    if player['id'] == target_uuid:
                        if best_team is not None:
                            player['team'] = 0 if best_team == 1 else 1
                            count_team_updates += 1
            
            count_updates += 1

        print(f"{count_updates} players' team information updated.")
        print(f"   - Team updates: {count_team_updates}")

        self.actions.sort(key=lambda a: a['timestamp'])

        scenario_data = {
            'version': 2,
            'scenes': self.scenes,
            'actions': self.actions
        }

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(scenario_data, f)
            print(f"Scenario saved: {filepath} ({len(self.scenes)} scenes, {len(self.actions)} actions)")
        except Exception as e:
            print(f"Save error: {e}")