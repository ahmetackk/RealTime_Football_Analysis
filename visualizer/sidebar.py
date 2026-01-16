import cv2
import numpy as np
import time
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

class SidebarVisualizer:
    def __init__(self, base_width, base_height, config=None, display_mode='stream'):
        self.display_mode = display_mode
        self.TARGET_HEIGHT = 1080
        self.TARGET_TOTAL_WIDTH = 1920
        self.PANEL_WIDTH = 400
        
        aspect_ratio = base_width / base_height
        self.height = min(base_height, self.TARGET_HEIGHT)
        video_width_from_height = int(self.height * aspect_ratio)
        
        max_video_width = self.TARGET_TOTAL_WIDTH - self.PANEL_WIDTH
        
        if video_width_from_height > max_video_width:
            self.width = max_video_width
            self.height = int(self.width / aspect_ratio)
        else:
            self.width = video_width_from_height
        
        self.total_width = self.width + self.PANEL_WIDTH
        self.PANEL_BG_COLOR = (30, 30, 30)
        self.config = config
        
        self.RADAR_MAX_WIDTH = min(300, self.PANEL_WIDTH - 20)
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.scale_line1 = 0.60
        self.scale_line2 = 0.45
        self.thick_line1 = 2
        self.thick_line2 = 1
        
        self.HEADER_Y = 30
        self.LINE_Y = 40
        self.LOG_START_Y = 80
        self.LOG_STEP_Y = 60
        
        self.event_log = []
        self.team_colors = {1: (235, 206, 135), 2: (203, 192, 255)}
        
        # --- OPTİMİZASYON 1: Statik Arka Planı Önbellekle ---
        self.static_background = np.full((self.height, self.total_width, 3), self.PANEL_BG_COLOR, dtype=np.uint8)
        self._draw_static_elements()
        
        # --- OPTİMİZASYON 2: Radar Sahasını Önbellekle ---
        self.cached_pitch = None
        if self.config:
            self.cached_pitch = draw_pitch(config=self.config)

    def _draw_static_elements(self):
        """Değişmeyen UI elemanlarını bir kez çizer."""
        base_x = self.width + 20
        cv2.putText(self.static_background, "LIVE ACTION LOG", (base_x, self.HEADER_Y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 200), 1)
        cv2.line(self.static_background, (base_x, self.LINE_Y), (self.total_width - 20, self.LINE_Y), (100, 100, 100), 1)

        estimated_radar_height = int(self.RADAR_MAX_WIDTH / 1.6)
        radar_bottom = self.height - 50
        ly = radar_bottom - estimated_radar_height - 30
        
        l_scale = 0.32
        # Bu yazılar hiç değişmiyor, her frame çizmeye gerek yok
        cv2.putText(self.static_background, ">85%", (base_x, ly), self.font, l_scale, (57, 255, 20), 1)
        cv2.putText(self.static_background, "65-85%", (base_x+50, ly), self.font, l_scale, (0, 255, 0), 1)
        cv2.putText(self.static_background, "45-65%", (base_x+115, ly), self.font, l_scale, (0, 165, 255), 1)
        cv2.putText(self.static_background, "30-45%", (base_x+180, ly), self.font, l_scale, (0, 100, 255), 1)
        cv2.putText(self.static_background, "<30%", (base_x+245, ly), self.font, l_scale, (0, 0, 255), 1)

    def update_team_colors(self, team1_bgr, team2_bgr):
        def brighten_color(bgr, factor=1.3):
            return tuple(min(255, int(c * factor)) for c in bgr)
        self.team_colors[1] = brighten_color(team1_bgr)
        self.team_colors[2] = brighten_color(team2_bgr) 

    def add_event(self, action_name, team_id, player_info, score):
        if score < 0.15: return
        
        current_time = time.time()
        for ev in self.event_log[:3]:
            if (ev['action_raw'] == action_name and ev['team_id'] == team_id and (current_time - ev['timestamp']) < 1.0):
                return

        team_color = self.team_colors.get(team_id, (200, 200, 200))
        team_str = "HOME" if team_id == 1 else "AWAY" if team_id == 2 else ""
        
        if score >= 0.85: conf_color = (57, 255, 20)
        elif score >= 0.65: conf_color = (0, 255, 0)
        elif score >= 0.45: conf_color = (0, 165, 255)
        elif score >= 0.30: conf_color = (0, 100, 255)
        else: conf_color = (0, 0, 255)

        text_team = f"{team_str} {player_info}" if team_str else player_info

        event_data = {
            'line1': f"{action_name.upper()}",
            'text_team': text_team,
            'text_conf': f" (Conf: {score:.2f})",
            'team_color': team_color,
            'conf_color': conf_color,
            'timestamp': current_time,
            'action_raw': action_name,
            'team_id': team_id
        }
        
        self.event_log.insert(0, event_data)
        if len(self.event_log) > 8:
            self.event_log.pop()

    def draw(self, frame):
        # OPTİMİZASYON: Her frame yeni canvas yaratma, static arkaplanı kopyala (Çok Hızlı)
        canvas = self.static_background.copy()
        
        frame_h, frame_w = frame.shape[:2]
        
        # Resize işleminden kaçış yok ama en azından doğru boyutlardaysa atlayabiliriz
        # Genelde videonun boyutu değişmez, burayı da optimize edebiliriz ama şimdilik kalsın.
        if frame_h > self.height or frame_w > self.width:
             # Basit resize mantığı (Aspect ratio koruyarak)
             scale = min(self.width/frame_w, self.height/frame_h)
             new_w = int(frame_w * scale)
             new_h = int(frame_h * scale)
             resized_frame = cv2.resize(frame, (new_w, new_h))
        else:
             resized_frame = frame
             new_h, new_w = frame_h, frame_w
        
        y_offset = (self.height - new_h) // 2
        x_offset = (self.width - new_w) // 2
        
        # Videoyu canvasa yerleştir
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

        # Sadece değişen yazıları (LOG) çiz
        base_x = self.width + 20
        estimated_radar_height = int(self.RADAR_MAX_WIDTH / 1.6)
        radar_bottom = self.height - 50
        ly = radar_bottom - estimated_radar_height - 30
        max_log_y = ly - 20
        current_y = self.LOG_START_Y
        
        for event in self.event_log:
            cv2.rectangle(canvas, (base_x - 10, current_y - 20), (base_x - 5, current_y + 25), event['conf_color'], -1)
            cv2.putText(canvas, event['line1'], (base_x + 10, current_y),
                        self.font, self.scale_line1, (255, 255, 255), self.thick_line1, cv2.LINE_AA)

            x_cursor = base_x + 10
            y_pos_line2 = current_y + 25

            cv2.putText(canvas, event['text_team'], (x_cursor, y_pos_line2),
                        self.font, self.scale_line2, event['team_color'], self.thick_line2, cv2.LINE_AA)
            
            (text_w, _), _ = cv2.getTextSize(event['text_team'], self.font, self.scale_line2, self.thick_line2)
            x_cursor += text_w

            cv2.putText(canvas, event['text_conf'], (x_cursor, y_pos_line2),
                        self.font, self.scale_line2, event['conf_color'], self.thick_line2, cv2.LINE_AA)

            current_y += self.LOG_STEP_Y
            if current_y + self.LOG_STEP_Y > max_log_y:
                break

        return canvas
    
    def draw_radar(self, detections, ball_detections, view_transformer, colors):
        if view_transformer is None or self.cached_pitch is None:
            return None
        
        # OPTİMİZASYON: Sahayı yeniden çizme, kopyala
        radar_img = self.cached_pitch.copy()
        
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        transformed_xy = view_transformer.transform_points(points=xy)

        for class_idx, color in enumerate(colors):
            mask = (detections.class_id == class_idx)
            if np.any(mask):
                radar_img = draw_points_on_pitch(
                    config=self.config, xy=transformed_xy[mask],
                    face_color=color, edge_color=sv.Color.BLACK, radius=14, pitch=radar_img
                )
        
        if len(ball_detections) > 0:
            ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            radar_ball = view_transformer.transform_points(points=ball_xy)
            radar_img = draw_points_on_pitch(
                config=self.config, xy=radar_ball, face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK, radius=8, pitch=radar_img
            )
        
        # Resize işlemini en sona bırakmak doğru
        original_h, original_w = radar_img.shape[:2]
        original_aspect_ratio = original_w / original_h
        
        radar_width = self.RADAR_MAX_WIDTH
        radar_height = int(radar_width / original_aspect_ratio)
        
        radar_img = cv2.resize(radar_img, (radar_width, radar_height))
        return radar_img
    
    def draw_with_radar(self, frame, detections, ball_detections, view_transformer, colors):
        canvas = self.draw(frame)
        
        radar_img = self.draw_radar(detections, ball_detections, view_transformer, colors)
        if radar_img is not None:
            radar_h, radar_w = radar_img.shape[:2]
            radar_y = self.height - radar_h - 50
            radar_x = self.width + (self.PANEL_WIDTH - radar_w) // 2
            
            # Numpy slicing çok hızlıdır, burası sorun değil
            canvas[radar_y:radar_y+radar_h, radar_x:radar_x+radar_w] = radar_img
        
        return canvas