import torch
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Modules from the repository
from utils.TAAD_Dataset import TAAD_Dataset
from models.model_TAAD_baseline import X3D_TAAD_Baseline

# --- SETTINGS ---
MODEL_PATH = "models/action_recognition.pt"
OUTPUT_FOLDER = "final_demo_detailed_videos_all"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Will be loaded from config
MAX_CLIPS_PER_CLASS = 5  # Default

# Target Unknown Classes
TARGET_UNKNOWN_CLASSES = [3, 4, 6, 7, 8] 
KNOWN_TRAINED_IDS = [1, 2, 5]

CLASS_NAMES_MAP = {
    0: 'Background', 1: 'Drive', 2: 'Pass', 3: 'Cross', 
    4: 'Throw-in', 5: 'Shot', 6: 'Header', 7: 'Tackle', 8: 'Block'
}

def unnormalize_frame(frame_tensor):
    """Converts a normalized tensor to a displayable uint8 image."""
    img = frame_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.45, 0.45, 0.45])
    std = np.array([0.225, 0.225, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1) * 255
    return img.astype(np.uint8)

def draw_text_with_bg(img, text, pos, font_scale=0.5, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    """Draws text with a background rectangle for better readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - text_h - 4), (x + text_w + 4, y + 4), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def create_video_clip(x, roi, masks, filename, info_lines):
    """Generates an MP4 video clip with bounding boxes and overlay text."""
    x = x.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    roi = roi.squeeze(0).cpu().numpy()
    masks = masks.squeeze(0).cpu().numpy()
    
    T, H, W, C = x.shape
    mean = np.array([0.45, 0.45, 0.45])
    std = np.array([0.225, 0.225, 0.225])
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 10.0, (W, H))
    
    for t in range(T):
        frame = x[t] * std + mean
        frame = np.clip(frame, 0, 1) * 255
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Draw Bounding Boxes
        num_players = roi.shape[0]
        for m in range(num_players):
            if masks[m, t] > 0.5:
                box = roi[m, t]
                x1, y1, x2, y2 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Overlay Information
        start_y = 30
        for i, (text, color) in enumerate(info_lines):
            draw_text_with_bg(frame, text, (10, start_y + (i * 20)), 
                              text_color=color, bg_color=(0, 0, 0, 128))
        
        out.write(frame)
    out.release()

def main():
    global MAX_CLIPS_PER_CLASS
    
    print("="*60)
    print("STAGE 2: KNOWN-UNKNOWN CLIP GENERATION")
    print("="*60)
    
    # Load speed config
    import json
    config_file = "speed_config.json"
    config = {"clip_max_per_class": 5, "unsupervised_samples": 100}  # Default
    
    if os.path.exists(config_file):
        print(f"[OK] Found config: {os.path.abspath(config_file)}")
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        print(f"[WARNING] Config not found at: {os.path.abspath(config_file)}")
        print(f"  Using defaults")
    
    MAX_CLIPS_PER_CLASS = config.get("clip_max_per_class", 5)
    
    print(f"Speed Profile: {config.get('name', 'Custom')}")
    print(f"   Max clips per class: {MAX_CLIPS_PER_CLASS}")

    # Load Model
    print("[1/3] Loading model...")
    
    model = X3D_TAAD_Baseline().to(DEVICE)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Model loading error: {e}")
        return
    model.eval()

    # Load Dataset
    print("[2/3] Loading dataset...")
    val_dataset = TAAD_Dataset("./", set_status='val')
    
    # Use config for dataset sampling
    dataset_limit = config.get("unsupervised_samples", 100)
    subset_indices = list(range(min(dataset_limit, len(val_dataset))))
    val_dataset_demo = Subset(val_dataset, subset_indices)
    loader = DataLoader(val_dataset_demo, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"[INFO] Using {len(subset_indices)} clips (out of {len(val_dataset)})")
    print(f"[DEMO] Max {MAX_CLIPS_PER_CLASS} clips per class")

    # Prepare Folders
    print("[3/3] Preparing output folders...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for cid in TARGET_UNKNOWN_CLASSES:
        cls_name = CLASS_NAMES_MAP[cid]
        os.makedirs(os.path.join(OUTPUT_FOLDER, cls_name), exist_ok=True)

    # Processing Loop
    print("\n[Processing clips...]")
    
    saved_counts = {k: 0 for k in TARGET_UNKNOWN_CLASSES}
    
    pbar = tqdm(total=len(loader))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            pbar.update(1)
            
            x, roi, masks, _, label = batch
            true_label_id = torch.max(label).item()
            
            # Skip if not target class
            if true_label_id not in TARGET_UNKNOWN_CLASSES: 
                continue
            
            # DEMO MODE: Skip if enough clips for this class
            if saved_counts[true_label_id] >= MAX_CLIPS_PER_CLASS:
                continue

            # Move to device
            x = x.to(DEVICE).float()
            roi = roi.to(DEVICE).float()
            masks = masks.to(DEVICE).float()

            # Model inference
            logits, embedding = model([x, roi, masks], return_embedding=True)
            
            # Supervised probabilities
            probs = torch.softmax(logits, dim=1)
            avg_probs = probs.mean(dim=(2, 3)).squeeze()
            top3_probs, top3_idxs = torch.topk(avg_probs, k=3)
            
            sup_details = []
            for i in range(3):
                cls_name = CLASS_NAMES_MAP.get(top3_idxs[i].item(), 'Unknown')
                prob_val = top3_probs[i].item()
                sup_details.append(f"{cls_name}({prob_val:.2f})")
            sup_text_str = ", ".join(sup_details)

            # Get ground truth name
            true_name = CLASS_NAMES_MAP.get(true_label_id, "Unknown")
            
            # Video generation - Simple: Show Ground Truth and Supervised predictions
            info_lines = [
                (f"Action: {true_name}", (0, 255, 0)),  # Green - Ground Truth
                (f"Model Top-3: {sup_text_str}", (200, 200, 200))  # Gray - Supervised
            ]
            
            file_name = f"{OUTPUT_FOLDER}/{true_name}/{true_name}_{saved_counts[true_label_id] + 1}.mp4"
            
            create_video_clip(x, roi, masks, file_name, info_lines)
            
            saved_counts[true_label_id] += 1
            pbar.set_description(f"Total: {sum(saved_counts.values())}")
            
            # Early exit if all classes have enough clips
            if all(saved_counts[k] >= MAX_CLIPS_PER_CLASS for k in TARGET_UNKNOWN_CLASSES):
                print("\n[DEMO] Demo limit reached for all classes!")
                break

    print("\n--- PROCESSING COMPLETE ---")
    print("Summary:")
    for cls_id in TARGET_UNKNOWN_CLASSES:
        print(f"   {CLASS_NAMES_MAP[cls_id]}: {saved_counts[cls_id]} clips")
    print(f"\n[DONE] Clips saved to '{OUTPUT_FOLDER}'")
    print("[OK] Stage 2 Complete!")

if __name__ == "__main__":
    main()