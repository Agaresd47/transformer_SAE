import torch
from transformers import BertTokenizer
from models.model.transformer import Transformer
from train import DialogTransformer
import numpy as np

class DialogPredictor:
    def __init__(self, model_path, device=None):
        # Set device
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load act and emotion mappings
        self.act_labels = {i: str(i) for i in range(11118)}
        self.emotion_labels = {i: str(i) for i in range(11118)}
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path):
        # Initialize model with same parameters as training
        model = DialogTransformer(
            d_model=768,
            n_heads=8,
            n_layers=6,
            drop_prob=0.1,
            device=self.device,
            num_act_classes=5,
            num_emotion_classes=7
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Filter out keys that do not match
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_state_dict}
        
        # Load the filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        
        return model

    def predict(self, text, feature_adjustments=None):
        """
        Predict act and emotion for a given text input and analyze SAE activations
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move inputs to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get predictions and SAE activations
        with torch.no_grad():
            # Get transformer hidden states
            hidden_states = self.model.transformer.encoder(
                input_ids, 
                attention_mask.unsqueeze(1).unsqueeze(2)
            )
            
            # Get SAE activations
            reconstructed, kl_div, encoded = self.model.sparse_autoencoder(hidden_states)
            
            # Get act and emotion predictions
            act_output, emotion_output = self.model(input_ids, attention_mask)[:2]
            
            # Get predicted classes
            act_pred = act_output.argmax(1).item()
            emotion_pred = emotion_output.argmax(1).item()
            
            # Get prediction probabilities
            act_probs = torch.softmax(act_output, dim=1)[0]
            emotion_probs = torch.softmax(emotion_output, dim=1)[0]
                    # Analyze SAE activations
        encoded_features = encoded.cpu().numpy()
        
        # Debug: Print the shape of encoded features
        #print(f"Encoded features shape: {encoded_features.shape}")
        
        # Get the number of features
        num_features = encoded_features.shape[-1]
        
        # Debug: Print the number of features
        #print(f"Number of features: {num_features}")
        
        # Apply feature adjustments if provided
        if feature_adjustments:
            for feature_idx, adjustment in feature_adjustments.items():
                if 0 <= feature_idx < num_features:
                    # Debug: Print the feature index and adjustment factor
                    #print(f"Adjusting feature {feature_idx} by factor {adjustment}")
                    # Debug: Print feature values before adjustment
                    #print(f"Feature {feature_idx} values before adjustment: {encoded_features[:, :, feature_idx]}")
                    encoded_features[:, :, feature_idx] *= adjustment
                    # Debug: Print feature values after adjustment
                    #print(f"Feature {feature_idx} values after adjustment: {encoded_features[:, :, feature_idx]}")
                else:
                    print(f"Warning: Feature index {feature_idx} is out of bounds and will be ignored.")
        
        # Get top 5 most activated features
        # Reshape to (batch_size * seq_length, feature_dim)
        feature_activations = encoded_features.reshape(-1, encoded_features.shape[-1])
        mean_activations = np.mean(np.abs(feature_activations), axis=0)
        
        # Debug: Print mean activations
        #print(f"Mean activations: {mean_activations}")
        
        top_features_idx = np.argsort(mean_activations)[-5:][::-1]
        top_features_values = mean_activations[top_features_idx]
        
        # Debug: Print top features
        #print(f"Top features indices: {top_features_idx}")
        #print(f"Top features values: {top_features_values}")
        
        return {
            'act': {
                'label': self.act_labels[act_pred],
                'confidence': float(act_probs[act_pred])
            },
            'emotion': {
                'label': self.emotion_labels[emotion_pred],
                'confidence': float(emotion_probs[emotion_pred])
            },
            'sae_analysis': {
                'top_features': [{
                    'feature_idx': int(idx),
                    'activation': float(val),
                } for idx, val in zip(top_features_idx, top_features_values)],
                'kl_divergence': float(kl_div.cpu().numpy()),
                'total_activation': float(np.mean(np.abs(feature_activations)))
            }
        }
    
def predict(model_path, text, feature_adjustments):
    # Initialize predictor
    predictor = DialogPredictor(model_path=model_path)
    
    # Perform prediction with featudfdre adjustments
    result = predictor.predict(text, feature_adjustments)
    
    # Print results
    print(f"\nInput text: {text}")
    print(f"\nPredictions:")
    print(f"Act: {result['act']['label']} (confidence: {result['act']['confidence']:.2f})")
    print(f"Emotion: {result['emotion']['label']} (confidence: {result['emotion']['confidence']:.2f})")
    
    print("\nSAE Analysis:")
    print(f"KL Divergence: {result['sae_analysis']['kl_divergence']:.4f}")
    print(f"Average Total Activation: {result['sae_analysis']['total_activation']:.4f}")
    print("\nTop 5 Most Activated Features:")
    for i, feature in enumerate(result['sae_analysis']['top_features'], 1):
        print(f"{i}. Feature {feature['feature_idx']}: {feature['activation']:.4f}")



if __name__ == "__main__":
    # Example usage
    model_path = r'saved/try2/model-3693.032.pt'  # Use your best model checkpoint
    text = "anger"
    feature_adjustments = {50: -11, 285: -11}  # Example adjustments
    
    predict(model_path, text, feature_adjustments)