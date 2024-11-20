import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_param=0.05, beta=3):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.sparsity_param = sparsity_param
        self.beta = beta
        
        # For analysis
        self.activation_history = []
        self.loss_history = []
        self.sparsity_history = []
        
    def forward(self, x):
        encoded = torch.sigmoid(self.encoder(x))
        decoded = self.decoder(encoded)
        
        # Store activations for analysis
        self.activation_history.append(encoded.detach().cpu().mean(0).numpy())
        
        rho_hat = torch.mean(encoded, dim=0)
        kl_div = self.kl_divergence(rho_hat)
        
        return decoded, kl_div, encoded
    
    def kl_divergence(self, rho_hat):
        rho = torch.full_like(rho_hat, self.sparsity_param)
        return torch.sum(rho * torch.log(rho/rho_hat) + 
                        (1-rho) * torch.log((1-rho)/(1-rho_hat)))
    
    def analyze_activations(self):
        """Analyze neuron activation patterns"""
        activations = np.array(self.activation_history)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(activations.T, cmap='viridis')
        plt.title('Neuron Activation Patterns Over Time')
        plt.xlabel('Training Steps')
        plt.ylabel('Neuron Index')
        plt.show()
        
    def analyze_sparsity(self):
        """Analyze sparsity levels"""
        activations = np.array(self.activation_history)
        sparsity = (activations < 0.1).mean(axis=1)
        
        plt.figure(figsize=(10, 5))
        plt.plot(sparsity)
        plt.title('Sparsity Level Over Time')
        plt.xlabel('Training Steps')
        plt.ylabel('Sparsity (% neurons < 0.1)')
        plt.show()
        
    def visualize_features(self):
        """Visualize learned features"""
        weights = self.encoder.weight.detach().cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        for i in range(min(16, weights.shape[0])):
            plt.subplot(4, 4, i+1)
            plt.imshow(weights[i].reshape(int(np.sqrt(weights.shape[1])), -1))
            plt.axis('off')
        plt.suptitle('Learned Features')
        plt.show() 