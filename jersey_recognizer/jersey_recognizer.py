import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision import models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parseq_path = os.path.join(BASE_DIR, "str", "parseq")
if parseq_path not in sys.path:
    sys.path.append(parseq_path)

try:
    from strhub.models.parseq.system import PARSeq
    from strhub.data.module import SceneTextDataModule
except ImportError:
    print(f"HATA: PARSeq modülleri yüklenemedi. Python path: {sys.path}")
    raise

_original_torch_load = torch.load
def _force_unsafe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _force_unsafe_load

class LegibilityClassifier34(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_ft = models.resnet34(weights=None)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1)
        
    def forward(self, x):
        x = self.model_ft(x)
        x = torch.sigmoid(x)
        return x

class JerseyNumberRecognizer:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Initializing JerseyNumberRecognizer ({self.device})...")

        self.legibility_model = LegibilityClassifier34()
        leg_path = os.path.join(BASE_DIR, 'models', 'legibility_resnet34.pth')
        
        if os.path.exists(leg_path):
            try:
                checkpoint = torch.load(leg_path, map_location=self.device, weights_only=False)
                if hasattr(checkpoint, '_metadata'): del checkpoint._metadata
                self.legibility_model.load_state_dict(checkpoint, strict=False)
                self.legibility_model.to(self.device)
                self.legibility_model.eval()
            except Exception as e:
                print(f"ERROR: Problem loading legibility model: {e}")
        else:
            print(f"WARNING: {leg_path} not found! (Legibility check may be disabled)")

        self.leg_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print("Loading YOLOv8-Pose...")
        yolo_path = 'yolov8n-pose.pt'
        if not os.path.exists(yolo_path):
             yolo_path = 'yolov8n-pose.pt'
        
        self.pose_model = YOLO(yolo_path)
        self.pose_model.overrides['save'] = False
        self.pose_model.overrides['save_txt'] = False
        self.pose_model.overrides['save_conf'] = False

        print("Loading PARSeq...")
        str_path = os.path.join(BASE_DIR, 'models', 'parseq_soccernet.ckpt')
        
        if os.path.exists(str_path):
            try:
                ckpt = torch.load(str_path, map_location=self.device, weights_only=False)
                hparams = ckpt.get('hyper_parameters', {})
                hparams.pop('monitor', None) 
                self.str_model = PARSeq(**hparams)
                
                state_dict = ckpt['state_dict']
                new_state_dict = {}
                for k, v in state_dict.items():
                    if not k.startswith('model.'):
                        new_key = 'model.' + k
                    else:
                        new_key = k
                    new_state_dict[new_key] = v
                
                self.str_model.load_state_dict(new_state_dict, strict=True)
                self.str_model.to(self.device)
                self.str_model.eval()
                
                self.str_transform = SceneTextDataModule.get_transform(self.str_model.hparams.img_size)
                print("System successfully initialized!")
            except Exception as e:
                 print(f"ERROR: STR model could not be loaded: {e}")
        else:
            print(f"ERROR: {str_path} not found!")

    def _crop_torso(self, img, keypoints, padding=5):
        if keypoints is None or len(keypoints) == 0: return None
        
        kpts = keypoints[0].data[0].cpu().numpy()
        relevant_indices = [5, 6, 11, 12] 
        relevant_pts = kpts[relevant_indices]

        if relevant_pts.shape[1] > 2:
            if np.any(relevant_pts[:, 2] < 0.3): return None
        
        xs = relevant_pts[:, 0]
        ys = relevant_pts[:, 1]
        valid_mask = (xs > 0) & (ys > 0)
        if not np.any(valid_mask): return None
        
        xs = xs[valid_mask]
        ys = ys[valid_mask]

        x_min = max(0, int(np.min(xs) - padding))
        x_max = min(img.shape[1], int(np.max(xs) + padding))
        y_min = max(0, int(np.min(ys) - padding))
        y_max = min(img.shape[0], int(np.max(ys)))

        if x_max <= x_min or y_max <= y_min: return None
        return img[y_min:y_max, x_min:x_max]

    def predict(self, player_img_numpy):
        if player_img_numpy is None or player_img_numpy.size == 0:
            return {'number': None, 'confidence': 0.0, 'is_legible': False}
            
        player_img_rgb = cv2.cvtColor(player_img_numpy, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(player_img_rgb)

        with torch.no_grad():
            leg_in = self.leg_transform(pil_img).unsqueeze(0).to(self.device)
            leg_score = self.legibility_model(leg_in).item()
        
        if leg_score < 0.5:
            return {'number': None, 'confidence': 0.0, 'is_legible': False}

        pose_results = self.pose_model.predict(player_img_rgb, verbose=False, device=self.device, save=False)
        
        if pose_results and pose_results[0].keypoints is not None:
            torso_crop = self._crop_torso(player_img_numpy, pose_results[0].keypoints)
        else:
            torso_crop = None

        if torso_crop is None:
            final_img = pil_img
        else:
            final_img = Image.fromarray(cv2.cvtColor(torso_crop, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            str_in = self.str_transform(final_img).unsqueeze(0).to(self.device)
            logits = self.str_model(str_in)
            probs = logits.softmax(-1)
            preds, probs = self.str_model.tokenizer.decode(probs)
            
            number = preds[0]
            confidence = probs[0].cpu().numpy().prod()

        return {
            'number': number,
            'confidence': float(confidence),
            'is_legible': True,
            'crop': torso_crop
        }