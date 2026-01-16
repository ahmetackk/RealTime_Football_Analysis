import torch
import numpy as np
import cv2
import pickle
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from tqdm import tqdm

# Modules from the repository
from utils.TAAD_Dataset import TAAD_Dataset
from models.model_TAAD_baseline import X3D_TAAD_Baseline

# --- SETTINGS ---
MODEL_PATH = "models/action_recognition.pt"
DISCOVERY_STATE_PATH = "discovery_state.pkl"
OUTPUT_FOLDER = "final_demo_detailed_videos_all" # Output folder for all videos
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Target Unknown Classes (Corrected IDs)
# 3: Cross, 4: Throw-in, 6: Header, 7: Tackle, 8: Block
TARGET_UNKNOWN_CLASSES = [3, 4, 6, 7, 8] 

# Known Classes (Corrected IDs)
# 1: Drive, 2: Pass, 5: Shot
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
        
        # Draw Bounding Boxes (Only for valid tracklets)
        num_players = roi.shape[0]
        for m in range(num_players):
            if masks[m, t] > 0.5:
                box = roi[m, t]
                x1, y1, x2, y2 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
                # Yellow bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Overlay Information Lines
        start_y = 30
        for i, (text, color) in enumerate(info_lines):
            draw_text_with_bg(frame, text, (10, start_y + (i * 20)), 
                              text_color=color, bg_color=(0, 0, 0, 128)) # Semi-transparent black background
        
        out.write(frame)
    out.release()

def main():
    print("--- FOOTPASS FULL ARCHIVE VIDEO GENERATOR ---")
    
    if not os.path.exists(DISCOVERY_STATE_PATH):
        print(f"ERROR: '{DISCOVERY_STATE_PATH}' not found! Please run the unsupervised discovery script first.")
        return

    # 1. Load Discovery Engine
    print("[1/4] Loading models and discovery state...")
    with open(DISCOVERY_STATE_PATH, 'rb') as f:
        state = pickle.load(f)
    
    pca = state['pca_model']
    kmeans = state['kmeans_model']
    mapping = state['cluster_mapping']
    
    # 2. Load Supervised Model
    model = X3D_TAAD_Baseline().to(DEVICE)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Model loading error: {e}")
        return
    model.eval()

    # 3. Load Dataset
    print("[2/4] Loading dataset...")
    # Shuffle=False to process videos in order (optional, but good for consistency)
    val_dataset = TAAD_Dataset("./", set_status='val') 
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 4. Prepare Folders
    print("[3/4] Preparing output folders...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for cid in TARGET_UNKNOWN_CLASSES:
        cls_name = CLASS_NAMES_MAP[cid]
        os.makedirs(os.path.join(OUTPUT_FOLDER, cls_name), exist_ok=True)

    # 5. Processing Loop
    print("[4/4] Starting analysis (Processing ALL target videos)...")
    
    saved_counts = {k: 0 for k in TARGET_UNKNOWN_CLASSES}
    
    pbar = tqdm(total=len(loader))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            pbar.update(1)
            
            x, roi, masks, _, label = batch
            
            # Identify the dominant label in the clip
            true_label_id = torch.max(label).item()
            
            # Skip if it's not a target unknown class
            if true_label_id not in TARGET_UNKNOWN_CLASSES: 
                continue

            # Move to device
            x = x.to(DEVICE).float()
            roi = roi.to(DEVICE).float()
            masks = masks.to(DEVICE).float()

            # --- MODEL INFERENCE ---
            logits, embedding = model([x, roi, masks], return_embedding=True)
            
            # 1. Supervised Analysis: Detailed Probabilities (Top-3)
            probs = torch.softmax(logits, dim=1)
            avg_probs = probs.mean(dim=(2, 3)).squeeze() # (Class,)
            
            top3_probs, top3_idxs = torch.topk(avg_probs, k=3)
            
            # Generate Supervised Detail Text (e.g., "Sup: BG(0.85), Pass(0.12), Shot(0.03)")
            sup_details = []
            for i in range(3):
                cls_name = CLASS_NAMES_MAP.get(top3_idxs[i].item(), 'Unknown')
                prob_val = top3_probs[i].item()
                sup_details.append(f"{cls_name}({prob_val:.2f})")
            sup_text_str = ", ".join(sup_details)

            # 2. Unsupervised Analysis: Clustering
            emb_time_avg = embedding.mean(dim=2)
            mask_time_avg = masks.mean(dim=2)
            valid_indices = (mask_time_avg > 0.5).nonzero(as_tuple=True)
            
            if len(valid_indices[0]) > 0:
                scene_feat = emb_time_avg[valid_indices].mean(dim=0)
            else:
                scene_feat = emb_time_avg.mean(dim=(0, 1))

            embed_vec = scene_feat.cpu().numpy().reshape(1, -1)
            embed_norm = normalize(embed_vec, norm='l2')
            embed_pca = pca.transform(embed_norm)
            cluster_id = kmeans.predict(embed_pca)[0]
            
            discovered_label_id = mapping.get(cluster_id, -1)
            discovered_name = CLASS_NAMES_MAP.get(discovered_label_id, f"Cluster {cluster_id}")
            true_name = CLASS_NAMES_MAP.get(true_label_id, "Unknown")

            # --- VIDEO GENERATION ---
            status_color = (0, 255, 0) if discovered_label_id == true_label_id else (0, 0, 255) # Green for Success, Red for Mismatch
            status_text = "SUCCESS" if discovered_label_id == true_label_id else "MISMATCH"
            
            # Information lines to be overlaid on the video
            info_lines = [
                (f"AI DISCOVERY: {discovered_name} ({status_text})", status_color),
                (f"Ground Truth: {true_name}", (255, 255, 255)),
                (f"Sup Top-3: {sup_text_str}", (200, 200, 200)) 
            ]
            
            file_name = f"{OUTPUT_FOLDER}/{true_name}/{true_name}_{saved_counts[true_label_id] + 1}.mp4"
            
            create_video_clip(x, roi, masks, file_name, info_lines)
            
            saved_counts[true_label_id] += 1
            pbar.set_description(f"Total Saved: {sum(saved_counts.values())}")

    print("\n--- PROCESSING COMPLETE ---")
    print("Summary of saved videos:")
    for cls_id in TARGET_UNKNOWN_CLASSES:
        print(f"   {CLASS_NAMES_MAP[cls_id]}: {saved_counts[cls_id]} videos")
    print(f"\nAll videos have been saved to '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    main()