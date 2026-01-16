import pickle
import torch
import numpy as np
from sklearn.preprocessing import normalize

class DiscoveryEngine:
    """Unsupervised action discovery using PCA + K-means clustering"""
    def __init__(self, state_path='discovery_state.pkl'):
        print(f"Loading Discovery Engine from {state_path}...")
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        
        # Load pre-trained PCA and K-means models for embedding clustering
        self.pca = state['pca_model']
        self.kmeans = state['kmeans_model']
        self.mapping = state['cluster_mapping']
        self.known_classes = state.get('known_classes', [1, 2, 5]) 
        
        self.class_names = {
            0: 'Background', 1: 'Drive', 2: 'Pass', 3: 'Cross', 
            4: 'Throw-in', 5: 'Shot', 6: 'Header', 7: 'Tackle', 8: 'Block'
        }

    def predict(self, embedding_tensor):
        """Predict action class from embedding: normalize -> PCA -> K-means -> label mapping"""
        # Handle multi-dimensional tensors by averaging spatial dimensions
        if isinstance(embedding_tensor, torch.Tensor):
            if embedding_tensor.dim() > 2:
                embedding_tensor = embedding_tensor.mean(dim=[2, 3, 4]) if embedding_tensor.dim() == 5 else embedding_tensor.mean(dim=2)
            
            vec = embedding_tensor.detach().cpu().numpy()
        else:
            vec = embedding_tensor

        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
            
        # L2 normalization before PCA
        vec_norm = normalize(vec, norm='l2')
        
        # Dimensionality reduction and clustering
        vec_pca = self.pca.transform(vec_norm)
        cluster_id = self.kmeans.predict(vec_pca)[0]
        
        # Map cluster to action label
        mapped_label_id = self.mapping.get(cluster_id, -1)
        label_name = self.class_names.get(mapped_label_id, f"Cluster-{cluster_id}")
        
        return mapped_label_id, label_name