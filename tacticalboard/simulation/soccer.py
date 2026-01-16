from dataclasses import dataclass, field
from typing import List, Tuple
import bisect
import supervision as sv
import cv2
import numpy as np 
import time 

from tacticalboard.configs.soccer import SoccerPitchConfiguration
from tacticalboard.annotators.soccer import draw_pitch, draw_player_on_pitch, draw_points_on_pitch

@dataclass(order=True)
class Position:
    timestamp: float = field(compare=True)
    x: int
    y: int

@dataclass
class Action:
    timestamp: float
    action_type: str
    text: str
    duration: float = 3.0

class Player:
    MAX_INTERPOLATION_GAP = 10.0 
    _auto_id_counter = 0
    
    def __init__(self, team_id: int, jersey_number: int, positions: List[Tuple[float, int, int]], class_id: int = 2, player_id: str = None, tracker_id: int = None):
        self.team_id = int(team_id) if team_id is not None else 0
        try:
            self.jersey_number = int(jersey_number) if jersey_number is not None else 0
        except (ValueError, TypeError):
            self.jersey_number = 0
        self.class_id = int(class_id) if class_id is not None else 2
        self.player_id = player_id
        self.tracker_id = tracker_id
        self.positions = sorted([Position(ts, x, y) for ts, x, y in positions])
        if not self.positions: raise ValueError("Player must have at least one position.")
        self._timestamps = [p.timestamp for p in self.positions]
    
    @property
    def display_number(self) -> str:
        if self.jersey_number > 0:
            return str(self.jersey_number)
        if self.tracker_id is not None:
            return f"ID{self.tracker_id}"
        return ""

    def get_position(self, current_time: float) -> Tuple[int, int] | None:
        if current_time < self.positions[0].timestamp: return None
        if current_time > self.positions[-1].timestamp: return None
        idx = bisect.bisect_right(self._timestamps, current_time)
        if idx > 0 and self._timestamps[idx - 1] == current_time:
            return self.positions[idx - 1].x, self.positions[idx - 1].y
        if idx == len(self.positions): 
             return self.positions[-1].x, self.positions[-1].y
             
        prev_pos = self.positions[idx - 1]
        next_pos = self.positions[idx]
        time_gap = next_pos.timestamp - prev_pos.timestamp
        if time_gap > self.MAX_INTERPOLATION_GAP: return None
        current_time_diff = current_time - prev_pos.timestamp
        if time_gap == 0: return prev_pos.x, prev_pos.y
        t = current_time_diff / time_gap
        x = prev_pos.x + (next_pos.x - prev_pos.x) * t
        y = prev_pos.y + (next_pos.y - prev_pos.y) * t
        return int(x), int(y)

class Ball:
    MAX_INTERPOLATION_GAP = 10.0
    def __init__(self, positions: List[Tuple[float, int, int]]):
        self.positions = sorted([Position(ts, x, y) for ts, x, y in positions])
        if not self.positions: raise ValueError("Ball must have at least one position.")
        self._timestamps = [p.timestamp for p in self.positions]
        
    def get_position(self, current_time: float) -> Tuple[int, int] | None:
        if current_time < self.positions[0].timestamp: return None
        if current_time > self.positions[-1].timestamp: return None
        idx = bisect.bisect_right(self._timestamps, current_time)
        if idx > 0 and self._timestamps[idx - 1] == current_time:
            return self.positions[idx - 1].x, self.positions[idx - 1].y
        if idx == len(self.positions):
             return self.positions[-1].x, self.positions[-1].y
        
        prev_pos = self.positions[idx - 1]
        next_pos = self.positions[idx]
        time_gap = next_pos.timestamp - prev_pos.timestamp
        if time_gap > self.MAX_INTERPOLATION_GAP: return None
        current_time_diff = current_time - prev_pos.timestamp
        if time_gap == 0: return prev_pos.x, prev_pos.y
        t = current_time_diff / time_gap
        x = prev_pos.x + (next_pos.x - prev_pos.x) * t
        y = prev_pos.y + (next_pos.y - prev_pos.y) * t
        return int(x), int(y)


class Simulation:
    """
    Manages the OpenCV simulation window, draws players, and advances time.
    """
    CONFIG = SoccerPitchConfiguration()
    TEAM_COLORS = {
        0: sv.Color.BLUE,
        1: sv.Color.RED,
        9: sv.Color.YELLOW,
    }
    CLASS_COLORS = {
        1: sv.Color.from_hex('#FF8C00'),
        3: sv.Color.YELLOW,
    }
    FRAME_RATE = 25
    PLAYBACK_SPEED = 1.0
    
    # Action colors (BGR format)
    ACTION_COLORS = {
        'goal': (0, 255, 0),       # Green
        'shot': (0, 165, 255),     # Orange
        'pass': (255, 255, 0),     # Cyan
        'tackle': (0, 0, 255),     # Red
        'save': (255, 0, 255),     # Magenta
        'foul': (0, 0, 200),       # Dark Red
        'corner': (255, 200, 0),   # Light Blue
        'offside': (128, 128, 255),# Light Red
        'default': (255, 255, 255) # White
    }

    def __init__(self, ball: Ball, players: List[Player], actions: List[Action] = None):
        self.ball = ball
        self.players = players
        self.actions = actions if actions else []
        self.min_time = float('inf')
        self.max_time = float('-inf')
        
        all_entities = players + ([ball] if ball else [])
        for entity in all_entities:
            if entity.positions:
                self.min_time = min(self.min_time, entity.positions[0].timestamp)
                self.max_time = max(self.max_time, entity.positions[-1].timestamp)
        
        if self.min_time == float('inf'):
            print("No data found to play.")
            self.min_time = 0
            self.max_time = 0
    
    def get_active_actions(self, current_time: float) -> List[Action]:
        active = []
        for action in self.actions:
            start_time = action.timestamp
            end_time = action.timestamp + action.duration
            
            if start_time <= current_time < end_time:
                active.append(action)
        return active
    
    TEAM_ACTION_COLORS = {
        0: (255, 100, 100),
        1: (100, 100, 255),
        -1: (255, 255, 255),
    }
    
    def draw_actions_on_pitch(self, pitch: np.ndarray, actions: List[Action]) -> np.ndarray:
        if not actions:
            return pitch
        
        y_offset = 60
        
        for action in actions:
            text = action.text
            
            team_id = -1
            if text.startswith("Team "):
                try:
                    team_id = int(text.split()[1])
                except (ValueError, IndexError):
                    team_id = -1
            
            color = self.TEAM_ACTION_COLORS.get(team_id, self.TEAM_ACTION_COLORS[-1])
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            x_pos = 10
            padding = 5
            overlay = pitch.copy()
            cv2.rectangle(overlay, 
                         (x_pos - padding, y_offset - text_height - padding),
                         (x_pos + text_width + padding, y_offset + padding),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, pitch, 0.4, 0, pitch)
            
            cv2.putText(pitch, text, (x_pos, y_offset), font, font_scale, color, thickness)
            
            y_offset += text_height + 15
        
        return pitch
            
    def run(self):
        """Starts the main simulation loop, synced to real-time at target FPS."""
        if self.min_time == self.max_time:
            print("Simulation duration is zero. Nothing to play.")
            return

        start_real_time = time.monotonic() 
        frame_duration_real = 1.0 / self.FRAME_RATE 
        next_frame_real_time = start_real_time 
        
        print(f"Simulation starting... Time: {self.min_time:.2f}s -> {self.max_time:.2f}s")
        print(f"Playback Speed: {self.PLAYBACK_SPEED}x, Target FPS: {self.FRAME_RATE}")
        print("Press 'q' in the simulation window to quit.")

        while True: 
            current_real_time = time.monotonic()
            
            elapsed_real_time = current_real_time - start_real_time
            current_simulation_time = self.min_time + (elapsed_real_time * self.PLAYBACK_SPEED)

            if current_simulation_time >= self.max_time: 
                 break

            frame_drawn = False
            if current_real_time >= next_frame_real_time:
                pitch = draw_pitch(config=self.CONFIG)
                
                for player in self.players:
                    pos = player.get_position(current_simulation_time) 
                    if pos is not None:
                        x, y = pos
                        
                        if player.class_id == 3:
                            color = self.CLASS_COLORS.get(3, sv.Color.YELLOW)
                        elif player.class_id == 1:
                            color = self.CLASS_COLORS.get(1, sv.Color.from_hex('#FF8C00'))
                        else:
                            color = self.TEAM_COLORS.get(player.team_id, sv.Color.WHITE)
                        
                        display_num = player.jersey_number if player.jersey_number > 0 else 0
                        pitch = draw_player_on_pitch(
                            config=self.CONFIG, xy=(x,y), jersey_number=display_num,
                            face_color=color, edge_color=sv.Color.BLACK, radius=16, pitch=pitch,
                            display_text=player.display_number
                        )
                
                ball_pos = self.ball.get_position(current_simulation_time) 
                if ball_pos is not None:
                    x, y = ball_pos
                    pitch = draw_points_on_pitch(
                        config=self.CONFIG, xy=[(x,y)], face_color=sv.Color.WHITE,
                        edge_color=sv.Color.BLACK, radius=10, pitch=pitch
                    )
                
                minutes = int(current_simulation_time / 60)
                seconds = int(current_simulation_time % 60)
                time_text = f"Time: {minutes:02}:{seconds:02}"
                cv2.putText(pitch, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 255, 255), 1)
                
                active_actions = self.get_active_actions(current_simulation_time)
                if active_actions:
                    pitch = self.draw_actions_on_pitch(pitch, active_actions)
                
                cv2.imshow("Soccer Simulation", pitch)
                frame_drawn = True

                next_frame_real_time += frame_duration_real

            key = cv2.waitKey(1) 
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        print("Simulation finished. Returning to Tactical Board.")