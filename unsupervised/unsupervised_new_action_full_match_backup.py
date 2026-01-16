import numpy as np
import os
import cv2
import h5py
from tqdm import tqdm
import torch
import pickle
from decord import VideoReader, cpu
from models.model_TAAD_baseline import X3D_TAAD_Baseline

# --- AYARLAR ---
FRAME_LIMIT = 7500   
STRIDE = 50          
CLIP_LENGTH = 50

def get_roi_masks(data, local_range):
    FRAME, PLAYER_ID, LEFT_TO_RIGHT, SHIRT_NUMBER, ROLE_ID, X_POS, Y_POS, X_SPEED, Y_SPEED, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, CLS = range(14)
    roles_list = [i for i in range(1,14)]
    all_rois, all_masks = [], []
    coeff = 1.125
    for left_to_right in [0,1]:
        for role_id in roles_list:
            tracklet_roi, tracklet_mask = [], []
            localdata = data[(data[:,LEFT_TO_RIGHT]==left_to_right)&(data[:,ROLE_ID]==role_id)].copy()
            for tidx, t in enumerate(local_range):
                bbox = localdata[(localdata[:,FRAME]==t)&(~np.isnan(localdata[:,ROI_X]))] if len(localdata) > 0 else []
                if len(bbox) > 0:
                    tlx = max(min(1920,int(bbox[0,ROI_X]-((coeff - 1.0)*bbox[0,ROI_WIDTH]//2))),0)
                    tly = max(min(1080,int(bbox[0,ROI_Y]-((coeff - 1.0)*bbox[0,ROI_HEIGHT]//2))),0)
                    brx = max(min(1920,int(bbox[0,ROI_X]+(coeff*bbox[0,ROI_WIDTH]))),0)
                    bry = max(min(1080,int(bbox[0,ROI_Y]+(coeff*bbox[0,ROI_HEIGHT]))),0)
                    # KoordinatlarÄ± 640x352 boyutuna gÃ¶re Ã¶lÃ§ekle
                    curr_roi = np.array([tidx, int(tlx/3), int(tly/3.068181), int(brx/3), int(bry/3.068181)])
                    tracklet_roi.append(curr_roi)
                    tracklet_mask.append(1.0)
                else:
                    tracklet_roi.append(np.array([tidx, 100*0.5, 100*0.5, 145*0.5, 198*0.5]))
                    tracklet_mask.append(0.0)
            all_rois.append(np.stack(tracklet_roi))
            all_masks.append(np.array(tracklet_mask))
    return np.stack(all_rois), np.stack(all_masks)

def get_clip(vr, kept_frame_range):
    frames = vr.get_batch(np.asarray(kept_frame_range, dtype=np.int64)).asnumpy()
    h, w, _ = frames[0].shape
    if w != 640 or h != 352:
        resized = [cv2.resize(fr, (640, 352), interpolation=cv2.INTER_AREA) for fr in frames]
        clip = np.stack(resized, axis=0)
    else:
        clip = frames
    clip = clip.astype(np.float32) / 255.0
    clip = (clip - 0.45) / 0.225
    return clip

def main():
    MODEL_CHECKPOINT = "models/action_recognition.pt"
    DATA_ROOT = "./"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = X3D_TAAD_Baseline().to(device)
    ckpt = torch.load(MODEL_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    h5file_path = os.path.join(DATA_ROOT, "data", 'val_tactical_data.h5')
    videopath = os.path.join(DATA_ROOT, "videos")
    
    with h5py.File(h5file_path, 'r') as f:
        game_idx_H = list(f.keys())[0]
    
    print(f"--- HIZLI TEST V3 (ROI FIX) BAÅžLIYOR: {game_idx_H} ---")
    
    gm_idx = game_idx_H.split('_')[1]
    vidfile = os.path.join(videopath, f'game_{gm_idx}.mp4')
    
    h5_file = h5py.File(h5file_path, 'r')
    data = h5_file[game_idx_H][:].astype(np.float64)
    data = data[~np.isnan(data[:, 9].astype(np.float64))].copy()
    
    total_frame_range = np.sort(np.unique(data[:, 0]))
    minf = int(total_frame_range.min())
    maxf = min(int(total_frame_range.max()), minf + FRAME_LIMIT)
    
    starts = np.arange(minf, maxf - CLIP_LENGTH, STRIDE)
    vr = VideoReader(vidfile, ctx=cpu(0))
    
    extracted_data = []

    with torch.no_grad():
        for i, sf in enumerate(tqdm(starts)):
            local_range = list(range(sf, sf + CLIP_LENGTH))
            try:
                clip_raw = get_clip(vr, local_range)
                clip_tensor = torch.from_numpy(clip_raw).float().permute(3, 0, 1, 2).unsqueeze(0).to(device)
            except:
                continue

            # ROI'leri burada hesaplÄ±yoruz
            rois, masks = get_roi_masks(data, local_range)
            
            # Model iÃ§in tensÃ¶r hazÄ±rlÄ±ÄŸÄ±
            M, T, _ = rois.shape
            rois_reshaped = rois.reshape(M, 1, T, 5)
            rois_reshaped[:,:,:,0] = rois_reshaped[:,:,:,0] - rois_reshaped[:,:,:1,0]
            rois_tensor = torch.from_numpy(rois_reshaped).float().to(device).permute(1, 0, 2, 3)
            masks_tensor = torch.from_numpy(masks).float().to(device).unsqueeze(0)

            logits, embedding = model([clip_tensor, rois_tensor, masks_tensor], return_embedding=True)
            
            # Feature Extraction (DoÄŸru Boyut: 512)
            emb_time_avg = embedding.mean(dim=2) 
            scene_feat = emb_time_avg.mean(dim=1).squeeze() 

            probs = torch.softmax(logits.mean(dim=(2,3)), dim=1).squeeze().cpu().numpy()
            predicted_class = np.argmax(probs)

            # --- DÃœZELTME: rois ve masks EKLENDÄ° ---
            extracted_data.append({
                'start_frame': sf,
                'embedding': scene_feat.cpu().numpy(),
                'probs': probs,
                'pred_class': predicted_class,
                'video_path': vidfile,
                'rois': rois,   # <-- Bu eksikti
                'masks': masks  # <-- Bu eksikti
            })

    output_file = "new_action_classes.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(extracted_data, f)
    
    print(f"\nTest verileri kaydedildi: {output_file}")
    print("ArtÄ±k ROI verileri mevcut!")

if __name__ == '__main__':
    main()