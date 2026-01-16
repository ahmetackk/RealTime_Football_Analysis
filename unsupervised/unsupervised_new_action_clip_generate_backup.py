import pickle
import numpy as np
import os
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from decord import VideoReader, cpu
from tqdm import tqdm

# --- AYARLAR ---
INPUT_FILE = "new_action_classes.pkl" # ROI iÃ§eren v3 dosyasÄ±
OUTPUT_ROOT = "NEW_ACTION_CLASSES"  # Ana Ã§Ä±ktÄ± klasÃ¶rÃ¼
NUM_CLUSTERS = 6                          # KÃ¼me sayÄ±sÄ±
MAX_VIDEOS_PER_CLUSTER = 50               # Her kÃ¼meden kaÃ§ video Ã¼retilsin? (Daha fazla Ã¶rnek istediÄŸin iÃ§in 50 yaptÄ±k)

CLASS_NAMES = {
    0: 'Background', 1: 'Drive', 2: 'Pass', 3: 'Cross', 
    4: 'Throw-in', 5: 'Shot', 6: 'Header', 7: 'Tackle', 8: 'Block'
}

def create_roi_video(meta, cluster_id, output_name):
    # Video dosyasÄ±nÄ± aÃ§
    vr = VideoReader(meta['video_path'], ctx=cpu(0))
    
    # Frame sÄ±nÄ±rlarÄ±nÄ± kontrol et
    if meta['start_frame'] + 50 >= len(vr):
        return

    # Kareleri oku
    frames = vr.get_batch(list(range(meta['start_frame'], meta['start_frame'] + 50))).asnumpy()
    
    # BoyutlandÄ±rma (Model 352x640 Ã§alÄ±ÅŸtÄ±ÄŸÄ± iÃ§in)
    H_target, W_target = 352, 640
    frames_resized = [cv2.resize(fr, (W_target, H_target)) for fr in frames]
    frames_resized = np.stack(frames_resized)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, 10.0, (W_target, H_target))
    
    # ROI ve Maskeleri al
    rois = meta.get('rois', None)
    masks = meta.get('masks', None)

    for t in range(50):
        frame = cv2.cvtColor(frames_resized[t], cv2.COLOR_RGB2BGR)
        
        # --- ROI Ã‡Ä°ZÄ°MÄ° ---
        if rois is not None and masks is not None:
            for p_idx in range(26):
                # Maske > 0.5 ise oyuncu aktiftir
                if masks[p_idx, t] > 0.5: 
                    box = rois[p_idx, t]
                    # KoordinatlarÄ± al (640x352'ye uygun)
                    x1, y1, x2, y2 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
                    
                    # TakÄ±m Renkleri (0-12: KÄ±rmÄ±zÄ±, 13-25: Mavi)
                    color = (0, 0, 255) if p_idx < 13 else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # --- BÄ°LGÄ° EKRANI ---
        # Ãœst siyah ÅŸerit
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
    print(f"--- TÃœM KÃœMELERÄ° Ä°NCELEME MODU (ROI AKTÄ°F) ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Hata: {INPUT_FILE} bulunamadÄ±. LÃ¼tfen extract v3 kodunu Ã§alÄ±ÅŸtÄ±rÄ±n.")
        return

    # 1. Veriyi YÃ¼kle
    print("Veri yÃ¼kleniyor...")
    with open(INPUT_FILE, 'rb') as f:
        data = pickle.load(f)
    
    X = np.array([d['embedding'] for d in data])
    print(f"Toplam Klip SayÄ±sÄ±: {len(X)}")
    
    # 2. KÃ¼meleme (K-Means)
    print("KÃ¼meleme yapÄ±lÄ±yor...")
    X_norm = normalize(X, norm='l2')
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_norm)
    
    # 3. KlasÃ¶r ve Video Ãœretimi
    print(f"Videolar '{OUTPUT_ROOT}' klasÃ¶rÃ¼ne hazÄ±rlanÄ±yor...")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    total_videos = 0
    
    # Her kÃ¼meyi tek tek dÃ¶n
    for cluster_id in range(NUM_CLUSTERS):
        # Bu kÃ¼meye ait kliplerin indekslerini bul
        indices = np.where(labels == cluster_id)[0]
        
        print(f" -> Cluster {cluster_id}: {len(indices)} adet klip bulundu.")
        
        # Alt klasÃ¶r oluÅŸtur
        cluster_folder = os.path.join(OUTPUT_ROOT, f"Cluster_{cluster_id}")
        os.makedirs(cluster_folder, exist_ok=True)
        
        count = 0
        # Klipleri iÅŸle (SÄ±nÄ±r koyarak)
        for idx in tqdm(indices, desc=f"Cluster {cluster_id} Processing"):
            if count >= MAX_VIDEOS_PER_CLUSTER:
                break
                
            vid_filename = os.path.join(cluster_folder, f"c{cluster_id}_vid_{count}.mp4")
            
            try:
                create_roi_video(data[idx], cluster_id, vid_filename)
                count += 1
            except Exception as e:
                print(f"Video hatasÄ± (Idx {idx}): {e}")
        
        total_videos += count

    print(f"\n--- Ä°ÅžLEM TAMAMLANDI ---")
    print(f"Toplam {total_videos} video oluÅŸturuldu.")
    print(f"Ã‡Ä±ktÄ± KlasÃ¶rÃ¼: {os.path.abspath(OUTPUT_ROOT)}")
    print("Ä°puÃ§larÄ±:")
    print(" - Cluster_X klasÃ¶rlerine girip videolarÄ± izle.")
    print(" - Hangi kÃ¼menin ne anlama geldiÄŸini (Ã–rn: Cluster 2 = Korner) not al.")

if __name__ == "__main__":
    main()