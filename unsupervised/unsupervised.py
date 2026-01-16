import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score, confusion_matrix, accuracy_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from utils.TAAD_Dataset import TAAD_Dataset
from models.model_TAAD_baseline import X3D_TAAD_Baseline
from tqdm import tqdm
import pandas as pd
import pickle
import os

# --- SETTINGS ---
MODEL_PATH = "models/action_recognition.pt"
DATA_ROOT = "./"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output directories
OUTPUT_DIR = "results"
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# CORRECTED CLASS MAPPING
CLASS_NAMES_MAP = {
    0: 'Background', 1: 'Drive', 2: 'Pass', 3: 'Cross', 
    4: 'Throw-in', 5: 'Shot', 6: 'Header', 7: 'Tackle', 8: 'Block'
}

KNOWN_CLASSES = [1, 2, 5] 

def extract_features(model, dataloader):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    print("Extracting features from dataset...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, roi, masks, _, labels = batch
            x = x.to(DEVICE).float()
            roi = roi.to(DEVICE).float()
            masks = masks.to(DEVICE).float()
            
            _, embeddings = model([x, roi, masks], return_embedding=True)
            
            valid_mask = masks.cpu() > 0.5
            valid_embeds = embeddings.cpu()[valid_mask]
            valid_labels = labels.reshape(-1)[valid_mask.reshape(-1)]
            
            all_embeddings.append(valid_embeds)
            all_labels.append(valid_labels)
            
    return torch.cat(all_embeddings).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

def map_clusters_to_labels(y_true, y_pred):
    """Maps K-Means clusters to real class IDs using the Hungarian Algorithm."""
    cm = contingency_matrix(y_true, y_pred)
    cost_matrix = cm.max() - cm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    unique_true_classes = np.unique(y_true)
    unique_pred_clusters = np.unique(y_pred)
    
    mapping = {}
    new_y_pred = np.zeros_like(y_pred)
    
    print("\n--- CLUSTER TO LABEL MAPPING TABLE ---")
    print(f"{'Cluster ID':<12} | {'Mapped Label':<15} | {'Class ID':<10}")
    print("-" * 45)
    
    for i in range(len(row_ind)):
        matrix_row_idx = row_ind[i]
        matrix_col_idx = col_ind[i]
        
        true_label = unique_true_classes[matrix_row_idx]
        cluster_label = unique_pred_clusters[matrix_col_idx]
        
        mapping[cluster_label] = true_label
        new_y_pred[y_pred == cluster_label] = true_label
        
        c_name = CLASS_NAMES_MAP.get(true_label, str(true_label))
        print(f"{cluster_label:<12} | {c_name:<15} | {true_label:<10}")
        
    return new_y_pred, mapping

def plot_confusion_matrix(cm, classes, title, filename, fmt='d', normalize=False):
    """Helper function for nice Confusion Matrix plots"""
    plt.figure(figsize=(10, 8))
    sns.set_style("white")
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = None, None

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                vmin=vmin, vmax=vmax, square=True, cbar_kws={'shrink': .8})
    
    plt.ylabel('Ground Truth Class', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Cluster (Mapped)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save to graphs directory
    output_path = os.path.join(GRAPHS_DIR, filename)
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot: {output_path}")
    plt.close()

def plot_distribution_comparison(y_true, y_pred, classes, filename):
    """Bar chart comparing Ground Truth counts vs Discovered counts"""
    true_counts = {c: np.sum(y_true == c_id) for c, c_id in zip(classes, np.unique(y_true))}
    pred_counts = {c: np.sum(y_pred == c_id) for c, c_id in zip(classes, np.unique(y_true))}
    
    data = []
    for cls in classes:
        cls_id = [k for k, v in CLASS_NAMES_MAP.items() if v == cls][0]
        data.append({'Class': cls, 'Count': true_counts.get(cls, 0), 'Type': 'Ground Truth'})
        data.append({'Class': cls, 'Count': pred_counts.get(cls, 0), 'Type': 'Discovered'})
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.barplot(data=df, x='Class', y='Count', hue='Type', palette='viridis', edgecolor='black')
    
    plt.title('Class Distribution: Ground Truth vs. AI Discovery', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xlabel('Action Class', fontsize=12)
    plt.legend(title='Data Source')
    plt.tight_layout()
    
    output_path = os.path.join(GRAPHS_DIR, filename)
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot: {output_path}")
    plt.close()

def simulate_discovery_final(embeddings, labels):
    print("\n--- OPEN WORLD DISCOVERY SIMULATION ---")
    
    # Filter Unknown Classes
    unknown_mask = ~np.isin(labels, KNOWN_CLASSES + [0])
    X_raw = embeddings[unknown_mask]
    y_true = labels[unknown_mask]
    
    print(f"Total Unknown Samples: {len(X_raw)}")
    unique_classes = np.unique(y_true)
    true_class_names = [CLASS_NAMES_MAP.get(c, str(c)) for c in unique_classes]
    print(f"Hidden Classes (Ground Truth): {true_class_names}")

    # 1. Pre-processing (L2 Norm + PCA)
    X_norm = normalize(X_raw, norm='l2')
    
    # ✅ Dynamic PCA components: min(50, n_samples - 1)
    n_samples = len(X_raw)
    n_components = min(50, n_samples - 1)
    
    if n_components < 10:
        print(f"[WARNING] Very few samples ({n_samples}), using {n_components} PCA components")
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_norm)
    print(f"PCA: {X_raw.shape} -> {X_pca.shape}")
    
    # 2. Clustering (K-Means)
    n_clusters = len(unique_classes)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    y_pred_raw = kmeans.fit_predict(X_pca)
    
    # 3. Mapping (Hungarian Algorithm)
    y_pred_matched, mapping_dict = map_clusters_to_labels(y_true, y_pred_raw)
    
    # 4. Metrics
    ari = adjusted_rand_score(y_true, y_pred_matched)
    acc = accuracy_score(y_true, y_pred_matched)
    
    print(f"\n" + "="*40)
    print(f"RESULTS REPORT")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Unsupervised Accuracy (ACC): {acc:.4f}")
    print(f"="*40 + "\n")
    
    # --- VISUALIZATION 1: Confusion Matrix (Counts) ---
    cm = confusion_matrix(y_true, y_pred_matched)
    plot_confusion_matrix(cm, true_class_names, 
                          f"Discovery Confusion Matrix (ARI: {ari:.3f})", 
                          "discovery_cm_counts.png", fmt='d')

    # --- VISUALIZATION 2: Confusion Matrix (Normalized %) ---
    plot_confusion_matrix(cm, true_class_names, 
                          f"Discovery Confusion Matrix (Normalized)", 
                          "discovery_cm_normalized.png", fmt='.2f', normalize=True)

    # --- VISUALIZATION 3: Class Distribution Comparison ---
    plot_distribution_comparison(y_true, y_pred_matched, true_class_names, 
                                 "discovery_class_distribution.png")
    
    # --- VISUALIZATION 4: t-SNE ---
    print("Generating t-SNE plot...")
    if len(X_pca) > 2000:
        idx = np.random.choice(len(X_pca), 2000, replace=False)
        X_vis = X_pca[idx]
        y_vis = y_true[idx]
    else:
        X_vis = X_pca
        y_vis = y_true
        
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_vis)
    
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], 
                    hue=[CLASS_NAMES_MAP.get(c, str(c)) for c in y_vis], 
                    palette="bright", s=80, alpha=0.8, edgecolor='k')
    plt.title(f"t-SNE Visualization of Discovered Classes (ARI: {ari:.3f})", fontsize=16)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="True Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_path = os.path.join(GRAPHS_DIR, 'discovery_tsne.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot: {output_path}")
    plt.close()

    # --- SAVE STATE ---
    print("\nSaving discovery state...")
    discovery_state = {
        'pca_model': pca,
        'kmeans_model': kmeans,
        'cluster_mapping': mapping_dict,
        'known_classes': KNOWN_CLASSES
    }
    
    state_path = os.path.join(OUTPUT_DIR, 'discovery_state.pkl')
    with open(state_path, 'wb') as f:
        pickle.dump(discovery_state, f)
    print(f"State saved to '{state_path}'")

if __name__ == "__main__":
    print("="*60)
    print("STAGE 1: UNSUPERVISED CLUSTERING")
    print("="*60)
    
    # ✅ Load speed config
    import json
    config_file = "speed_config.json"
    config = {"unsupervised_samples": 150}  # Default: medium
    
    if os.path.exists(config_file):
        print(f"[OK] Found config: {os.path.abspath(config_file)}")
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        print(f"[WARNING] Config not found at: {os.path.abspath(config_file)}")
        print(f"  Using defaults (medium profile)")
    
    SAMPLE_LIMIT = config.get("unsupervised_samples", 150)
    
    print(f"Speed Profile: {config.get('name', 'Medium (default)')}")
    print(f"   Samples: {SAMPLE_LIMIT}")
    
    print("Loading Model...")
    model = X3D_TAAD_Baseline().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Loading Dataset...")
    val_set = TAAD_Dataset(DATA_ROOT, set_status='val')
    
    # Apply sample limit
    from torch.utils.data import Subset
    subset_indices = list(range(min(SAMPLE_LIMIT, len(val_set))))
    val_set_demo = Subset(val_set, subset_indices)
    
    val_loader = DataLoader(val_set_demo, batch_size=4, shuffle=False, num_workers=2)
    
    print(f" Using {len(subset_indices)} samples (out of {len(val_set)} total)")
    
    embeds, labs = extract_features(model, val_loader)
    simulate_discovery_final(embeds, labs)
    
    print("\nStage 1 Complete!")
    print(f"Graphs saved to: {GRAPHS_DIR}")
    print(f"Discovery state saved to: {os.path.join(OUTPUT_DIR, 'discovery_state.pkl')}")