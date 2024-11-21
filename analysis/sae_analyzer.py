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
        if self.metadata is None or ('activations' not in self.metadata and 'encoded_features' not in self.metadata):
            print("Computing activations as metadata is not available...")
            activations = self.compute_activations()
        else:
            # Use either 'activations' or 'encoded_features'
            activations = self.metadata.get('activations', self.metadata.get('encoded_features'))
            
        if activations is None or activations.size == 0:
            print("No activations available to plot!")
            return
            
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
            # Use either 'activations' or 'encoded_features'
            activations = self.metadata.get('activations', self.metadata.get('encoded_features'))
            
        if activations is None:
            print("No activations available to plot!")
            return
            
        # Ensure activations are 2D (samples x features)
        if activations.ndim > 2:
            activations = activations.reshape(-1, activations.shape[-1])
        
        # Limit number of features
        num_features = min(num_features, activations.shape[1])
        feature_activations = activations[:, :num_features]
        
        # Calculate correlations
        corr_matrix = np.corrcoef(feature_activations.T)
        
        # Plot
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
    
    def plot_reconstruction_error(self):
        """Plot reconstruction error distribution"""
        if self.metadata is None or 'reconstruction_errors' not in self.metadata:
            print("Reconstruction errors not available in metadata")
            return
        
        errors = self.metadata['reconstruction_errors']
        
        # Flatten if needed
        if errors.ndim > 1:
            errors = errors.flatten()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, bins=50)
        plt.title('Distribution of Reconstruction Errors')
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Count')
    
    def get_most_active_features(self, top_k=10):
        """Return the indices of the most active features"""
        activations = self.compute_activations()
        mean_activations = np.mean(activations, axis=0)
        top_indices = np.argsort(mean_activations)[-top_k:]
        return top_indices, mean_activations[top_indices] 
    


    def get_dead_features(self):
        """Return indices of dead features (features that rarely activate)"""
        if self.metadata is None:
            activations = self.compute_activations()
        else:
            activations = self.metadata.get('activations', self.metadata.get('encoded_features'))
        
        if activations is None:
            print("No activations available!")
            return []
            
        # Ensure activations are 2D
        if activations.ndim > 2:
            activations = activations.reshape(-1, activations.shape[-1])
            
        # Consider a feature dead if it activates less than 1% of the time
        activation_rates = np.mean(np.abs(activations) > 1e-6, axis=0)
        dead_features = np.where(activation_rates < 0.01)[0]
        return dead_features

    def get_activation_rates(self):
        """Return activation rates for each feature"""
        if self.metadata is None:
            activations = self.compute_activations()
        else:
            activations = self.metadata.get('activations', self.metadata.get('encoded_features'))
        
        if activations is None:
            print("No activations available!")
            return np.array([])
            
        # Ensure activations are 2D
        if activations.ndim > 2:
            activations = activations.reshape(-1, activations.shape[-1])
            
        # Calculate activation rates (how often each feature is active)
        activation_rates = np.mean(np.abs(activations) > 1e-6, axis=0)
        return activation_rates

    def get_feature_statistics(self):
        """Return comprehensive statistics about feature activations"""
        if self.metadata is None:
            activations = self.compute_activations()
        else:
            activations = self.metadata.get('activations', self.metadata.get('encoded_features'))
        
        if activations is None:
            print("No activations available!")
            return {}
            
        # Ensure activations are 2D
        if activations.ndim > 2:
            activations = activations.reshape(-1, activations.shape[-1])
            
        stats = {
            'mean_activations': np.mean(activations, axis=0),
            'std_activations': np.std(activations, axis=0),
            'activation_rates': self.get_activation_rates(),
            'dead_features': self.get_dead_features(),
            'max_activations': np.max(activations, axis=0),
            'min_activations': np.min(activations, axis=0)
        }
        
        return stats