import cv2
import numpy as np
import time
import supervision as sv

class Visualizer:
    def __init__(self, config, display_mode='stream'):
        self.config = config
        self.display_mode = display_mode
        
        self.colors = [
            sv.Color.from_hex('#00BFFF'),
            sv.Color.from_hex('#FF1493'),
            sv.Color.from_hex('#FFD700'),
            sv.Color.from_hex('#FF4500')
        ]
        self.team_names = {1: "HOME", 2: "AWAY"}
        
        self.action_overlay_log = []
        self.team_colors_bgr = {
            1: (255, 191, 0),
            2: (147, 20, 255)
        }
        
        self._init_annotators()

    def _init_annotators(self):
        palette = sv.ColorPalette(colors=self.colors)
        
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=palette, 
            thickness=2
        )
        
        self.label_annotator = sv.LabelAnnotator(
            color=palette,
            text_color=sv.Color.from_hex('#000000'),
            text_position=sv.Position.TOP_CENTER,
            text_scale=0.3, 
            text_thickness=1,                
            text_padding=3,                         
            border_radius=5                          
        )
        
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'), base=15, height=12
        )
        
        # Vertex annotator kullanılmıyorsa kaldırdım, gerekirse ekleyebilirsiniz.

    def update_team_colors(self, team1_bgr, team2_bgr):
        def bgr_to_sv(bgr):
            return sv.Color(r=int(bgr[2]), g=int(bgr[1]), b=int(bgr[0]))

        self.colors[0] = bgr_to_sv(team1_bgr)
        self.colors[1] = bgr_to_sv(team2_bgr)
        
        def brighten_color(bgr, factor=1.3):
            return tuple(min(255, int(c * factor)) for c in bgr)
        self.team_colors_bgr[1] = brighten_color(team1_bgr)
        self.team_colors_bgr[2] = brighten_color(team2_bgr)
        
        self._init_annotators()
    
    def add_action_overlay(self, action_name, team_id, player_info, score):
        if score < 0.5:
            return
        
        current_time = time.time()
        
        for ev in self.action_overlay_log[:3]:
            if (ev['action'] == action_name and 
                ev['team_id'] == team_id and 
                (current_time - ev['timestamp']) < 2.0):
                return
        
        team_color = self.team_colors_bgr.get(team_id, (200, 200, 200))
        
        event_data = {
            'action': action_name,
            'player_info': player_info,
            'team_id': team_id,
            'team_color': team_color,
            'score': score,
            'timestamp': current_time
        }
        
        self.action_overlay_log.insert(0, event_data)
        if len(self.action_overlay_log) > 3:
            self.action_overlay_log.pop()

    def annotate_frame(self, frame, detections, ball_detections, player_stats, pitch_keypoints=None):
        # Frame kopyası burada 1 kere alınır, sonrasında hep bunun üzerine çizilir.
        annotated_frame = frame.copy()
        
        # Topu çiz
        annotated_frame = self.triangle_annotator.annotate(annotated_frame, ball_detections)
        
        labels = []
        # enumerate yerine zip kullanarak ufak bir hızlanma
        tracker_ids = detections.tracker_id
        class_ids = detections.class_id
        
        if tracker_ids is not None:
            for tracker_id, class_id in zip(tracker_ids, class_ids):
                if class_id == 2:
                    labels.append("REFEREE")
                    continue
                if class_id == 3:
                    labels.append("KEEPER")
                    continue
                
                final_text = f"ID {tracker_id}"
                
                if tracker_id in player_stats:
                    stats = player_stats[tracker_id]
                    team_prefix = ""
                    
                    if stats['team_locked']:
                        team_prefix = self.team_names.get(stats['final_team'], "")
                    
                    if stats['jersey_locked'] and stats['final_number'] is not None:
                        text_content = f"#{stats['final_number']}"
                        final_text = f"{team_prefix} {text_content}" if team_prefix else text_content
                    elif team_prefix:
                        final_text = f"{team_prefix} ID {tracker_id}"
                
                labels.append(final_text)

        annotated_frame = self.ellipse_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels=labels)
        
        # Action Overlay çizimi (Optimize edilmiş versiyonu çağırır)
        annotated_frame = self._draw_action_overlay(annotated_frame)
        
        return annotated_frame
    
    def _draw_action_overlay(self, frame):
        if not self.action_overlay_log:
            return frame
        
        current_time = time.time()
        # List comprehension yerine filter daha hızlı olabilir ama liste kısa olduğu için fark etmez.
        self.action_overlay_log = [ev for ev in self.action_overlay_log 
                                   if (current_time - ev['timestamp']) < 3.0]
        
        if not self.action_overlay_log:
            return frame

        frame_h, frame_w = frame.shape[:2]
        margin_right = 20
        margin_top = 35
        line_height = 50
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2
        
        # OPTİMİZASYON: Tüm frame üzerinde değil, sadece kutu alanında (ROI) işlem yap
        for i, event in enumerate(self.action_overlay_log[:3]):
            score = event['score']
            
            if score >= 0.85: conf_color = (57, 255, 20)
            elif score >= 0.70: conf_color = (0, 255, 0)
            elif score >= 0.55: conf_color = (0, 200, 255)
            else: conf_color = (0, 100, 255)
            
            action_text = f"{event['action'].upper()} {event['player_info']}"
            (text_w, text_h), _ = cv2.getTextSize(action_text, font, font_scale, thickness)
            
            x = frame_w - text_w - margin_right - 15
            y = margin_top + (i * line_height)
            
            # Koordinatları hesapla
            rect_x1 = x - 12
            rect_y1 = y - text_h - 8
            rect_x2 = frame_w - margin_right + 5
            rect_y2 = y + 12
            
            # Sınır kontrolü (Frame dışına taşmayı önle)
            rect_x1 = max(0, rect_x1)
            rect_y1 = max(0, rect_y1)
            rect_x2 = min(frame_w, rect_x2)
            rect_y2 = min(frame_h, rect_y2)

            if rect_x2 <= rect_x1 or rect_y2 <= rect_y1:
                continue

            # --- OPTİMİZASYON BAŞLANGICI ---
            # Sadece ilgili alanı (ROI) kesip al
            roi = frame[rect_y1:rect_y2, rect_x1:rect_x2]
            
            # Siyah kutu için kopyalama (sadece ROI kadar, tüm frame değil)
            overlay = roi.copy()
            cv2.rectangle(overlay, (0, 0), (rect_x2-rect_x1, rect_y2-rect_y1), (0, 0, 0), -1)
            
            # Şeffaflığı sadece bu küçük parçaya uygula
            cv2.addWeighted(overlay, 0.7, roi, 0.3, 0, roi)
            
            # Takım rengi çubuğu
            bar_w = 7 # Çubuk genişliği
            cv2.rectangle(roi, (0, 0), (bar_w, rect_y2-rect_y1), event['team_color'], -1)
            
            # Yazıyı yaz (global koordinatları ROI'ye çevirerek veya frame üzerine yazarak)
            # Frame üzerine yazmak daha net sonuç verir (ROI üzerine yazarsak anti-aliasing bozulabilir)
            # O yüzden ROI'yi geri yapıştırdıktan sonra frame'e yazıyoruz.
            frame[rect_y1:rect_y2, rect_x1:rect_x2] = roi
            
            cv2.putText(frame, action_text, (x, y), font, font_scale, conf_color, thickness, cv2.LINE_AA)
            # --- OPTİMİZASYON BİTİŞİ ---
        
        return frame