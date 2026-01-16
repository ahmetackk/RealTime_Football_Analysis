"""
UNSUPERVISED ACTION DISCOVERY SYSTEM
3-Stage Pipeline for Real Match Videos:
1. Extract embeddings from entire video
2. Cluster known-unknown actions (Cross, Throw-in, Header, Tackle)
3. Discover completely new actions (GK saves, diving headers, etc.)

Usage:
    python unsupervised_analyzer.py --video match.mp4 --output results/discovery --device cuda
"""

import os
import sys
import cv2
import torch
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict

# Visualization
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

# Local modules
from analyzer.analyzer import FootballAnalyzer
from sports.configs.soccer import SoccerPitchConfiguration
from action_recognizer.action_recognizer_optimized import ActionRecognizer


# ============================================================================
# CONFIGURATION
# ============================================================================

CLASS_NAMES_MAP = {
    0: 'Background',
    1: 'Drive', 
    2: 'Pass', 
    3: 'Cross',
    4: 'Throw-in',
    5: 'Shot',
    6: 'Header',
    7: 'Tackle',
    8: 'Block'
}

KNOWN_CLASSES = [1, 2, 5]  # Drive, Pass, Shot (trained)
KNOWN_UNKNOWN_CLASSES = [3, 4, 6, 7, 8]  # Cross, Throw-in, Header, Tackle, Block
SUPERVISED_CONF_THRESH = 0.70  # If confidence > 70% for known class, skip
OUTLIER_THRESHOLD = 1.0  # Distance threshold for completely new actions
GROUPING_THRESHOLD = 0.50  # Distance for dynamic clustering


# ============================================================================
# STAGE 1: EMBEDDING EXTRACTION
# ============================================================================

class EmbeddingExtractor:
    """Extract action embeddings from entire video using ActionRecognizer"""
    
    def __init__(self, action_recognizer, output_dir):
        self.recognizer = action_recognizer
        self.output_dir = output_dir
        
        self.embeddings_db = []  # List of {frame, player_id, embedding, action, confidence}
        
        # Initialize helpers (needed for analyzer.process_players)
        self._team_assigner = None
        self._jersey_model = None
    
    def _get_raw_detections(self, frame, analyzer):
        """
        Extract raw player detections from analyzer
        Returns list of [x1, y1, x2, y2, conf, class_id, tracker_id]
        """
        # Lazy init helpers
        if self._team_assigner is None:
            from team_assigner.team_assigner import TeamAssigner
            self._team_assigner = TeamAssigner()
        
        if self._jersey_model is None:
            from jersey_recognizer.jersey_recognizer import JerseyNumberRecognizer
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._jersey_model = JerseyNumberRecognizer(device=device)
        
        # Get player detections with tracking
        player_dets = analyzer.process_players(frame, self._team_assigner, self._jersey_model)
        
        # Convert to raw detection format [x1, y1, x2, y2, conf, class_id, tracker_id]
        raw_detections = []
        if player_dets and hasattr(player_dets, 'xyxy'):
            for i in range(len(player_dets.xyxy)):
                xyxy = player_dets.xyxy[i]
                conf = player_dets.confidence[i] if hasattr(player_dets, 'confidence') else 0.9
                cls_id = player_dets.class_id[i] if hasattr(player_dets, 'class_id') else 1
                tracker_id = player_dets.tracker_id[i] if hasattr(player_dets, 'tracker_id') else i
                
                raw_detections.append([
                    float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]),
                    float(conf), int(cls_id), int(tracker_id)
                ])
        
        return raw_detections
        
    def process_video(self, video_path, analyzer, progress_callback=None):
        """
        Process entire video and extract embeddings for each detected action
        """
        print("\n" + "="*60)
        print("STAGE 1: EMBEDDING EXTRACTION")
        print("="*60)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")
        
        frame_idx = 0
        
        with tqdm(total=total_frames, desc="Extracting embeddings") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get raw detections (player tracking)
                raw_detections = self._get_raw_detections(frame, analyzer)
                
                # Update action recognizer buffer
                self.recognizer.update(frame, raw_detections)
                
                # Get predictions with embeddings
                if frame_idx % self.recognizer.stride == 0:
                    result = self.recognizer.predict()
                    
                    if result and 'predictions' in result:
                        predictions = result['predictions']
                        
                        for player_id, action_data in predictions.items():
                            action_name = action_data['action']
                            confidence = action_data['score']
                            embedding = action_data.get('embedding')
                            
                            if embedding is not None:
                                # Save embedding entry
                                self.embeddings_db.append({
                                    'frame': frame_idx,
                                    'player_id': player_id,
                                    'embedding': embedding,
                                    'action': action_name,
                                    'confidence': confidence,
                                    'supervised_probs': action_data.get('probs', None)
                                })
                
                frame_idx += 1
                pbar.update(1)
                
                if progress_callback:
                    progress_callback(frame_idx, total_frames)
        
        cap.release()
        
        print(f"\n✓ Extracted {len(self.embeddings_db)} action embeddings")
        
        # Save embeddings to disk
        embeddings_file = os.path.join(self.output_dir, 'embeddings.pkl')
        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings_db, f)
        print(f"✓ Saved embeddings to: {embeddings_file}")
        
        return self.embeddings_db


# ============================================================================
# STAGE 2: KNOWN-UNKNOWN CLUSTERING
# ============================================================================

class KnownUnknownDiscovery:
    """
    Cluster embeddings to discover known-unknown classes
    (Cross, Throw-in, Header, Tackle, Block)
    """
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.pca = None
        self.kmeans = None
        self.cluster_mapping = {}
        
    def analyze(self, embeddings_db):
        """
        Perform clustering on unknown actions
        """
        print("\n" + "="*60)
        print("STAGE 2: KNOWN-UNKNOWN DISCOVERY")
        print("="*60)
        
        # Filter: Only confident non-known actions
        filtered_entries = []
        for entry in embeddings_db:
            # Skip background
            if 'background' in entry['action'].lower():
                continue
            
            # Skip known classes with high confidence
            is_known = any(CLASS_NAMES_MAP[k].lower() in entry['action'].lower() 
                          for k in KNOWN_CLASSES)
            if is_known and entry['confidence'] > SUPERVISED_CONF_THRESH:
                continue
            
            filtered_entries.append(entry)
        
        print(f"Candidates for clustering: {len(filtered_entries)}")
        
        if len(filtered_entries) < 10:
            print("⚠ Not enough samples for clustering!")
            return None
        
        # Extract embeddings
        embeddings = np.array([e['embedding'] for e in filtered_entries])
        
        # Preprocessing
        print("Preprocessing embeddings...")
        embeddings_norm = normalize(embeddings, norm='l2')
        
        # ✅ Dynamic PCA components (max 50, but not more than n_samples)
        n_samples = len(embeddings_norm)
        n_components = min(50, n_samples - 1)  # PCA needs n_components < n_samples
        
        print(f"PCA: {n_samples} samples → {n_components} components")
        
        self.pca = PCA(n_components=n_components, random_state=42)
        embeddings_pca = self.pca.fit_transform(embeddings_norm)
        
        # Clustering (assume 5 known-unknown classes)
        # ✅ Dynamic cluster count based on sample size
        max_clusters = len(KNOWN_UNKNOWN_CLASSES)  # 5
        n_clusters = min(max_clusters, max(2, n_samples // 5))  # At least 5 samples per cluster
        
        print(f"Running K-Means with {n_clusters} clusters (target: {max_clusters})...")
        
        if n_clusters < max_clusters:
            print(f"⚠ Warning: Not enough samples for {max_clusters} clusters, using {n_clusters}")
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_labels = self.kmeans.fit_predict(embeddings_pca)
        
        # Analyze cluster distributions
        print("\n--- CLUSTER DISTRIBUTION ---")
        for i in range(n_clusters):
            count = np.sum(cluster_labels == i)
            print(f"Cluster {i}: {count} samples")
        
        # Save discovery state
        discovery_state = {
            'pca_model': self.pca,
            'kmeans_model': self.kmeans,
            'cluster_mapping': {},  # Will be filled by manual inspection
            'known_classes': KNOWN_CLASSES
        }
        
        state_file = os.path.join(self.output_dir, 'discovery_state.pkl')
        with open(state_file, 'wb') as f:
            pickle.dump(discovery_state, f)
        print(f"✓ Saved discovery state to: {state_file}")
        
        # Visualization
        self._visualize_clusters(embeddings_pca, cluster_labels, filtered_entries)
        
        return {
            'entries': filtered_entries,
            'embeddings_pca': embeddings_pca,
            'cluster_labels': cluster_labels
        }
    
    def _visualize_clusters(self, embeddings_pca, labels, entries):
        """Generate t-SNE visualization"""
        print("\nGenerating t-SNE visualization...")
        
        # Sample if too many points
        if len(embeddings_pca) > 2000:
            idx = np.random.choice(len(embeddings_pca), 2000, replace=False)
            X_vis = embeddings_pca[idx]
            y_vis = labels[idx]
        else:
            X_vis = embeddings_pca
            y_vis = labels
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_embedded = tsne.fit_transform(X_vis)
        
        plt.figure(figsize=(12, 10))
        sns.set_style("whitegrid")
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                             c=y_vis, cmap='tab10', s=50, alpha=0.7, edgecolor='k')
        plt.colorbar(scatter, label='Cluster ID')
        plt.title("t-SNE Visualization of Action Clusters", fontsize=16)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.tight_layout()
        
        plot_file = os.path.join(self.output_dir, 'clusters_tsne.png')
        plt.savefig(plot_file, dpi=300)
        print(f"✓ Saved t-SNE plot: {plot_file}")
        plt.close()


# ============================================================================
# STAGE 3: CLIP GENERATION
# ============================================================================

class ClipGenerator:
    """Generate video clips for discovered actions"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.clips_dir = os.path.join(output_dir, 'discovered_clips')
        os.makedirs(self.clips_dir, exist_ok=True)
        
    def generate_clips(self, video_path, cluster_data):
        """
        Create video clips for each clustered action
        """
        print("\n" + "="*60)
        print("STAGE 3: CLIP GENERATION")
        print("="*60)
        
        if cluster_data is None:
            print("⚠ No cluster data available, skipping clip generation")
            return
        
        entries = cluster_data['entries']
        labels = cluster_data['cluster_labels']
        
        # Group by cluster
        clusters = defaultdict(list)
        for entry, label in zip(entries, labels):
            clusters[label].append(entry)
        
        # Create folder for each cluster
        for cluster_id, cluster_entries in clusters.items():
            cluster_dir = os.path.join(self.clips_dir, f'cluster_{cluster_id}')
            os.makedirs(cluster_dir, exist_ok=True)
            
            print(f"\nCluster {cluster_id}: {len(cluster_entries)} clips")
            
            # Generate clips (max 10 per cluster for demo)
            for idx, entry in enumerate(cluster_entries[:10]):
                self._extract_clip(video_path, entry, cluster_dir, idx)
        
        print(f"\n✓ Clips saved to: {self.clips_dir}")
    
    def _extract_clip(self, video_path, entry, output_dir, clip_idx):
        """Extract 3-second clip around the action"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        center_frame = entry['frame']
        start_frame = max(0, center_frame - int(fps * 1.5))  # 1.5 seconds before
        end_frame = center_frame + int(fps * 1.5)  # 1.5 seconds after
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        clip_filename = os.path.join(output_dir, 
                                     f'clip_{clip_idx}_frame{center_frame}_player{entry["player_id"]}.mp4')
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(clip_filename, 
                             cv2.VideoWriter_fourcc(*'mp4v'), 
                             fps, (width, height))
        
        for _ in range(int(end_frame - start_frame)):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add overlay text
            info_text = f"Action: {entry['action']} ({entry['confidence']:.2f})"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(frame)
        
        cap.release()
        out.release()


# ============================================================================
# STAGE 4: NEW ACTION DISCOVERY
# ============================================================================

class NewActionDiscovery:
    """Discover completely new action patterns"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.new_actions_dir = os.path.join(output_dir, 'new_discoveries')
        os.makedirs(self.new_actions_dir, exist_ok=True)
        
        self.new_classes = []
        self.next_id = 0
        
    def discover(self, embeddings_db, discovery_state_path):
        """
        Find actions that don't match any known cluster
        """
        print("\n" + "="*60)
        print("STAGE 4: NEW ACTION DISCOVERY")
        print("="*60)
        
        # Load discovery state
        with open(discovery_state_path, 'rb') as f:
            state = pickle.load(f)
        
        pca = state['pca_model']
        kmeans = state['kmeans_model']
        
        new_discoveries = []
        
        for entry in tqdm(embeddings_db, desc="Scanning for new actions"):
            # Skip known classes
            is_known = any(CLASS_NAMES_MAP[k].lower() in entry['action'].lower() 
                          for k in KNOWN_CLASSES)
            if is_known and entry['confidence'] > SUPERVISED_CONF_THRESH:
                continue
            
            # Skip background
            if 'background' in entry['action'].lower():
                continue
            
            # Transform embedding
            emb = entry['embedding'].reshape(1, -1)
            emb_norm = normalize(emb, norm='l2')
            emb_pca = pca.transform(emb_norm)
            
            # Check distance to known clusters
            distances = kmeans.transform(emb_pca)
            min_dist = np.min(distances)
            
            # If far from all clusters → NEW ACTION!
            if min_dist > OUTLIER_THRESHOLD:
                new_discoveries.append({
                    'entry': entry,
                    'distance': min_dist,
                    'embedding_pca': emb_pca[0]
                })
        
        print(f"\n✓ Found {len(new_discoveries)} potential new actions!")
        
        if len(new_discoveries) > 0:
            # Dynamic clustering of new discoveries
            self._cluster_new_actions(new_discoveries)
        
        return new_discoveries
    
    def _cluster_new_actions(self, discoveries):
        """Dynamically cluster new discoveries"""
        print("\nDynamic clustering of new actions...")
        
        for discovery in discoveries:
            emb = discovery['embedding_pca'].reshape(1, -1)
            emb_norm = normalize(emb, norm='l2')
            
            # Find closest new class
            best_dist = float('inf')
            best_idx = -1
            
            for i, cls in enumerate(self.new_classes):
                dist = np.linalg.norm(cls['center'] - emb_norm)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            # Assign to existing or create new
            if best_dist < GROUPING_THRESHOLD and best_idx != -1:
                # Add to existing class
                old_center = self.new_classes[best_idx]['center']
                count = self.new_classes[best_idx]['count']
                new_center = (old_center * count + emb_norm) / (count + 1)
                new_center = normalize(new_center, norm='l2')
                
                self.new_classes[best_idx]['center'] = new_center
                self.new_classes[best_idx]['count'] += 1
                self.new_classes[best_idx]['samples'].append(discovery)
            else:
                # Create new class
                self.new_classes.append({
                    'id': self.next_id,
                    'center': emb_norm,
                    'count': 1,
                    'samples': [discovery]
                })
                self.next_id += 1
        
        # Report
        print(f"\n--- NEW ACTION CLASSES ---")
        for cls in self.new_classes:
            print(f"New Class {cls['id']}: {cls['count']} samples")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_unsupervised_analysis(video_path, output_base_dir, analyzer, 
                               action_recognizer, progress_callback=None):
    """
    Complete 4-stage unsupervised analysis pipeline
    """
    print("\n" + "="*80)
    print("UNSUPERVISED ACTION DISCOVERY - FULL PIPELINE")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Output: {output_base_dir}")
    
    # Create output structure
    os.makedirs(output_base_dir, exist_ok=True)
    
    # STAGE 1: Extract embeddings
    extractor = EmbeddingExtractor(action_recognizer, output_base_dir)
    embeddings_db = extractor.process_video(video_path, analyzer, progress_callback)
    
    # STAGE 2: Cluster known-unknowns
    discovery = KnownUnknownDiscovery(output_base_dir)
    cluster_data = discovery.analyze(embeddings_db)
    
    # STAGE 3: Generate clips
    clip_gen = ClipGenerator(output_base_dir)
    clip_gen.generate_clips(video_path, cluster_data)
    
    # STAGE 4: Discover new actions
    new_discovery = NewActionDiscovery(output_base_dir)
    discovery_state_file = os.path.join(output_base_dir, 'discovery_state.pkl')
    new_actions = new_discovery.discover(embeddings_db, discovery_state_file)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"✓ Embeddings: {len(embeddings_db)}")
    print(f"✓ Clustered actions: {len(cluster_data['entries']) if cluster_data else 0}")
    print(f"✓ New discoveries: {len(new_actions)}")
    print(f"✓ Results saved to: {output_base_dir}")
    
    return {
        'embeddings': embeddings_db,
        'clusters': cluster_data,
        'new_actions': new_actions
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised Action Discovery Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--inference-size", type=int, default=960)
    parser.add_argument("--action-mode", default="unsupervised", choices=["supervised", "unsupervised"])
    parser.add_argument("--execution-mode", default="async", choices=["sync", "async"])
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("UNSUPERVISED ACTION DISCOVERY - STANDALONE MODE")
    print("="*80)
    print(f"Video: {args.video}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"Inference size: {args.inference_size}")
    print(f"Action mode: {args.action_mode}")
    print(f"Execution mode: {args.execution_mode}")
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Setup models
    MODELS = {
        'player': 'models/player_detection.engine',
        'pitch': 'models/pitch_detection.engine',
        'ball': 'models/ball_detection.engine'
    }
    
    config = SoccerPitchConfiguration()
    analyzer = FootballAnalyzer(MODELS, config, args.device, inference_size=args.inference_size)
    
    # Action recognizer
    action_recognizer = ActionRecognizer(
        model_path="models/action_recognition.pt",
        device=args.device,
        mode=args.action_mode,
        execution_mode=args.execution_mode
    )
    
    try:
        # Run pipeline
        results = run_unsupervised_analysis(
            video_path=args.video,
            output_base_dir=args.output,
            analyzer=analyzer,
            action_recognizer=action_recognizer,
            progress_callback=None
        )
        
        print("\n✓ Analysis complete!")
        print(f"✓ Results saved to: {args.output}")
        
        # Cleanup
        action_recognizer.shutdown()
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        action_recognizer.shutdown()
        sys.exit(1)