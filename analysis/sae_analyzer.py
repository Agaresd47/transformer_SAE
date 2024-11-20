import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class SAEAnalyzer:
    def __init__(self, model=None, data_iterator=None, device=None, metadata=None):
        self.model = model
        self.data_iterator = data_iterator
        self.device = device
        self.metadata = metadata
        
    @classmethod
    def from_metadata(cls, metadata):
        """Create analyzer instance from metadata"""
        return cls(metadata=metadata)
    
    def plot_activation_distribution(self):
        """Plot distribution of neuron activations"""
        if self.metadata is None:
            activations = self.compute_activations()
        else:
            activations = self.metadata['activations']
        
        plt.figure(figsize=(12, 6))
        sns.histplot(activations.flatten(), bins=50)
        plt.title('Distribution of Neuron Activations')
        plt.xlabel('Activation Value')
        plt.ylabel('Count')
        
    def plot_feature_correlations(self, num_features=50):
        """Plot correlation matrix of feature activations"""
        if self.metadata is None:
            activations = self.compute_activations()
        else:
            activations = self.metadata['activations']
            
        # Debug prints
        print(f"Initial activations shape: {activations.shape}")
        
        # If we only have one feature, we can't do correlation analysis
        if activations.shape[1] < 2:
            print("Not enough features for correlation analysis. Skipping plot.")
            return
        
        # Limit number of features
        num_features = min(num_features, activations.shape[1])
        
        feature_activations = activations[:, :num_features]
        print(f"Feature activations shape: {feature_activations.shape}")
        
        corr_matrix = np.corrcoef(feature_activations.T)
        print(f"Correlation matrix shape: {corr_matrix.shape}")
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
        plt.title(f'Feature Correlations (First {num_features} Features)')
        plt.tight_layout()
        
    def compute_activations(self, num_batches=10):
        """Compute and cache activations for analysis"""
        activations = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.data_iterator):
                if i >= num_batches:
                    break
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Ensure encoded output has proper dimensions
                _, _, encoded = self.model(input_ids, attention_mask)
                
                # Add dimensions if necessary
                if encoded.dim() == 0:
                    encoded = encoded.unsqueeze(0)  # Add batch dimension
                if encoded.dim() == 1:
                    encoded = encoded.unsqueeze(1)  # Add feature dimension
                    
                activations.append(encoded.cpu().numpy())
        
        if not activations:  # Check if we got any activations
            raise ValueError("No activations were collected. Check if the data iterator is empty.")
        
        try:
            self.cached_activations = np.concatenate(activations, axis=0)
        except ValueError as e:
            print("Debug info:")
            print(f"Number of activation batches: {len(activations)}")
            print(f"Shape of first activation batch: {activations[0].shape if activations else 'No activations'}")
            raise e
        
        return self.cached_activations
    
    def plot_reconstruction_error(self, num_batches=10):
        """Plot reconstruction error distribution"""
        if self.metadata is None:
            raise ValueError("No metadata available for reconstruction error analysis")
        
        errors = self.metadata['reconstruction_errors']
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors.flatten(), bins=50)
        plt.title('Distribution of Reconstruction Errors')
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Count')
    
    def get_most_active_features(self, top_k=10):
        """Return the indices of the most active features"""
        activations = self.compute_activations()
        mean_activations = np.mean(activations, axis=0)
        top_indices = np.argsort(mean_activations)[-top_k:]
        return top_indices, mean_activations[top_indices] 