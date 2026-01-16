from sklearn.cluster import KMeans
import numpy as np
import cv2

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.kmeans = None
        self.team_colors_lab = {}
        
    def _extract_jersey_region(self, player_crop):
        h, w = player_crop.shape[:2]
        
        MIN_SIZE = 40
        if h < MIN_SIZE or w < MIN_SIZE:
            scale = max(MIN_SIZE / h, MIN_SIZE / w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            player_crop = cv2.resize(player_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = new_h, new_w
        
        top = int(h * 0.15)
        bottom = int(h * 0.55)
        left = int(w * 0.15)
        right = int(w * 0.85)
        
        jersey_region = player_crop[top:bottom, left:right]
        
        if jersey_region.size == 0:
            return player_crop
        
        return jersey_region
    
    def _remove_grass_pixels(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        is_grass = (
            (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 80) &
            (hsv[:,:,1] >= 30) & (hsv[:,:,1] <= 180) &
            (hsv[:,:,2] >= 30) & (hsv[:,:,2] <= 200)
        )
        
        is_neon_green = (
            (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 90) &
            ((hsv[:,:,1] > 150) | (hsv[:,:,2] > 180))
        )
        
        actual_grass = is_grass & ~is_neon_green
        
        non_grass_pixels = image[~actual_grass]
        
        total_pixels = image.shape[0] * image.shape[1]
        non_grass_ratio = len(non_grass_pixels) / total_pixels if total_pixels > 0 else 0
        
        if non_grass_ratio < 0.15:
            return image.reshape(-1, 3), True
        
        return non_grass_pixels, False
    
    def _get_dominant_color_kmeans(self, pixels, n_clusters=2):
        if len(pixels) < n_clusters:
            return np.mean(pixels, axis=0) if len(pixels) > 0 else np.array([128, 128, 128])
        
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels = pixels[indices]
        
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=3, max_iter=10, random_state=42)
        kmeans.fit(pixels)
        
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_idx = labels[np.argmax(counts)]
        
        return kmeans.cluster_centers_[dominant_idx]
    
    def get_player_color(self, player_crop):
        if player_crop.size == 0:
            return np.array([128, 128, 128])
        
        jersey_region = self._extract_jersey_region(player_crop)
        
        valid_pixels, is_green_jersey = self._remove_grass_pixels(jersey_region)
        
        if len(valid_pixels) < 5:
            return np.array([128, 128, 128])
        
        dominant_color = self._get_dominant_color_kmeans(valid_pixels, n_clusters=2)
        
        return dominant_color
    
    def _bgr_to_lab(self, bgr_color):
        bgr_pixel = np.uint8([[bgr_color]])
        lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)[0][0]
        return lab_pixel.astype(np.float32)
    
    def _color_distance_lab(self, color1_lab, color2_lab):
        return np.sqrt(np.sum((color1_lab - color2_lab) ** 2))
    
    def fit_team_colors(self, player_crops):
        print(f"Learning team colors... ({len(player_crops)} crops)")
        
        if len(player_crops) < 10:
            print("Insufficient number of crops!")
            self.team_colors = {1: np.array([0, 0, 255]), 2: np.array([255, 0, 0])}
            return
        
        if len(player_crops) > 300:
            indices = np.random.choice(len(player_crops), 300, replace=False)
            player_crops = [player_crops[i] for i in indices]
        
        colors_bgr = []
        colors_lab = []
        
        for crop in player_crops:
            if crop.size == 0:
                continue
            
            color = self.get_player_color(crop)
            if color is not None and not np.isnan(color).any():
                colors_bgr.append(color)
                colors_lab.append(self._bgr_to_lab(color))
        
        if len(colors_lab) < 10:
            print("Insufficient valid colors!")
            self.team_colors = {1: np.array([0, 0, 255]), 2: np.array([255, 0, 0])}
            return
        
        colors_lab = np.array(colors_lab)
        colors_bgr = np.array(colors_bgr)
        
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(colors_lab)
        
        self.kmeans = kmeans
        
        labels = kmeans.labels_
        
        team1_mask = labels == 0
        team2_mask = labels == 1
        
        if np.sum(team1_mask) > 0:
            self.team_colors[1] = np.mean(colors_bgr[team1_mask], axis=0)
            self.team_colors_lab[1] = np.mean(colors_lab[team1_mask], axis=0)
        else:
            self.team_colors[1] = np.array([255, 255, 255])
            self.team_colors_lab[1] = self._bgr_to_lab(self.team_colors[1])
        
        if np.sum(team2_mask) > 0:
            self.team_colors[2] = np.mean(colors_bgr[team2_mask], axis=0)
            self.team_colors_lab[2] = np.mean(colors_lab[team2_mask], axis=0)
        else:
            self.team_colors[2] = np.array([0, 0, 0])
            self.team_colors_lab[2] = self._bgr_to_lab(self.team_colors[2])
        
        print(f"Team 1: {self.team_colors[1].astype(int)}")
        print(f"Team 2: {self.team_colors[2].astype(int)}")
    
    def get_player_team(self, player_crop):

        if self.kmeans is None:
            return 1
        
        if player_crop.size == 0:
            return 1
        
        h, w = player_crop.shape[:2]
        
        if h < 30 or w < 30:
            scale = max(30 / h, 30 / w)
            player_crop = cv2.resize(player_crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            h, w = player_crop.shape[:2]
        
        top = int(h * 0.15)
        bottom = int(h * 0.55)
        left = int(w * 0.15)
        right = int(w * 0.85)
        jersey = player_crop[top:bottom, left:right]
        
        if jersey.size == 0:
            return 1
        
        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        
        is_grass = (
            (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 80) &
            (hsv[:,:,1] >= 30) & (hsv[:,:,1] <= 180) &
            (hsv[:,:,2] >= 30) & (hsv[:,:,2] <= 200)
        )
        
        is_neon_green = (
            (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 90) &
            ((hsv[:,:,1] > 150) | (hsv[:,:,2] > 180))
        )
        
        actual_grass = is_grass & ~is_neon_green
        valid_pixels = jersey[~actual_grass]
        
        if len(valid_pixels) < jersey.shape[0] * jersey.shape[1] * 0.15:
            valid_pixels = jersey.reshape(-1, 3)
        
        if len(valid_pixels) < 5:
            return 1
        
        player_color_bgr = np.median(valid_pixels, axis=0)
        
        bgr_pixel = np.uint8([[player_color_bgr]])
        lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
        
        dist1 = np.sum((lab_pixel - self.team_colors_lab[1]) ** 2)
        dist2 = np.sum((lab_pixel - self.team_colors_lab[2]) ** 2)
        
        return 1 if dist1 < dist2 else 2
