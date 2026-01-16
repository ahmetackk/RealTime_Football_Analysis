import pickle
import numpy as np
import os
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from decord import VideoReader, cpu
from tqdm import tqdm

# --- SETTINGS ---
INPUT_FILE = "new_action_classes.pkl"
OUTPUT_ROOT = "NEW_ACTION_CLASSES"
NUM_CLUSTERS = 6
# ✅ Will be loaded from config
MAX_VIDEOS_PER_CLUSTER = 3  # Default

CLASS_NAMES = {
    0: 'Background', 1: 'Drive', 2: 'Pass', 3: 'Cross', 
    4: 'Throw-in', 5: 'Shot', 6: 'Header', 7: 'Tackle', 8: 'Block'
}

def create_roi_video(meta, cluster_id, output_name):
    vr = VideoReader(meta['video_path'], ctx=cpu(0))
    
    if meta['start_frame'] + 50 >= len(vr):
        return

    frames = vr.get_batch(list(range(meta['start_frame'], meta['start_frame'] + 50))).asnumpy()
    
    H_target, W_target = 352, 640
    frames_resized = [cv2.resize(fr, (W_target, H_target)) for fr in frames]
    frames_resized = np.stack(frames_resized)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, 10.0, (W_target, H_target))
    
    rois = meta.get('rois', None)
    masks = meta.get('masks', None)

    for t in range(50):
        frame = cv2.cvtColor(frames_resized[t], cv2.COLOR_RGB2BGR)
        
        # Draw ROIs
        if rois is not None and masks is not None:
            for p_idx in range(26):
                if masks[p_idx, t] > 0.5: 
                    box = rois[p_idx, t]
                    x1, y1, x2, y2 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
                    color = (0, 0, 255) if p_idx < 13 else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Info overlay
        cv2.rectangle(frame, (0, 0), (W_target, 50), (0,0,0), -1)
        
        model_pred_id = meta.get('pred_class', -1)
        model_pred_name = CLASS_NAMES.get(model_pred_id, "Unknown")
        
        text1 = f"CLUSTER: {cluster_id} | Frame: {meta['start_frame']}"
        text2 = f"Supervised Model Thinks: {model_pred_name}"
        
        cv2.putText(frame, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, text2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        out.write(frame)
    out.release()

def main():
    global MAX_VIDEOS_PER_CLUSTER
    
    print("="*60)
    print("STAGE 4: NEW ACTION CLIP GENERATION")
    print("="*60)
    
    # ✅ Load speed config
    import json
    config_file = "speed_config.json"
    config = {"new_action_videos_per_cluster": 3}  # Default
    
    if os.path.exists(config_file):
        print(f"[OK] Found config: {os.path.abspath(config_file)}")
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        print(f"[WARNING] Config not found at: {os.path.abspath(config_file)}")
        print(f"  Using defaults")
    
    MAX_VIDEOS_PER_CLUSTER = config.get("new_action_videos_per_cluster", 3)
    
    print(f"Speed Profile: {config.get('name', 'Custom')}")
    print(f"   Videos per cluster: {MAX_VIDEOS_PER_CLUSTER}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found!")
        return

    # Load data
    print("Loading data...")
    with open(INPUT_FILE, 'rb') as f:
        data = pickle.load(f)
    
    X = np.array([d['embedding'] for d in data])
    print(f"Total clips: {len(X)}")
    
    # Clustering
    print("Clustering...")
    X_norm = normalize(X, norm='l2')
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_norm)
    
    # Generate videos
    print(f"Generating {MAX_VIDEOS_PER_CLUSTER} videos per cluster...")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    total_videos = 0
    
    for cluster_id in range(NUM_CLUSTERS):
        indices = np.where(labels == cluster_id)[0]
        
        print(f" -> Cluster {cluster_id}: {len(indices)} clips found")
        
        cluster_folder = os.path.join(OUTPUT_ROOT, f"Cluster_{cluster_id}")
        os.makedirs(cluster_folder, exist_ok=True)
        
        count = 0
        for idx in tqdm(indices, desc=f"Cluster {cluster_id}"):
            if count >= MAX_VIDEOS_PER_CLUSTER:
                break
                
            vid_filename = os.path.join(cluster_folder, f"c{cluster_id}_vid_{count}.mp4")
            
            try:
                create_roi_video(data[idx], cluster_id, vid_filename)
                count += 1
            except Exception as e:
                print(f"Error (Idx {idx}): {e}")
        
        total_videos += count

    print(f"\nCreated {total_videos} videos")
    print(f"Output: {os.path.abspath(OUTPUT_ROOT)}")

if __name__ == "__main__":
    main()