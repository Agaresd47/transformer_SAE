import torch
from transformers import BertTokenizer
from models.model.transformer import Transformer
from train import DialogTransformer

class DialogPredictor:
    def __init__(self, model_path, device=None):
        # Set device
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load act and emotion mappings with all classes
        self.act_labels = {
            i: str(i) for i in range(11118)  # Match the number of classes from training
        }
        
        self.emotion_labels = {
            i: str(i) for i in range(11118)  # Match the number of classes from training
        }
        
        # Initialize model (after labels initialization)
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
            num_act_classes=len(self.act_labels),
            num_emotion_classes=len(self.emotion_labels)
        ).to(self.device)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def predict(self, text):
        """
        Predict act and emotion for a given text input
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
        
        # Get predictions
        with torch.no_grad():
            act_output, emotion_output = self.model(input_ids, attention_mask)
            
            # Get predicted classes
            act_pred = act_output.argmax(1).item()
            emotion_pred = emotion_output.argmax(1).item()
            
            # Get prediction probabilities
            act_probs = torch.softmax(act_output, dim=1)[0]
            emotion_probs = torch.softmax(emotion_output, dim=1)[0]
        
        return {
            'act': {
                'label': self.act_labels[act_pred],
                'confidence': float(act_probs[act_pred])
            },
            'emotion': {
                'label': self.emotion_labels[emotion_pred],
                'confidence': float(emotion_probs[emotion_pred])
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = DialogPredictor(
        model_path='saved/model-1.773.pt'  # Replace with your model path
    )
    
    # Example prediction
    text = "I'm so happy to see you today!"
    result = predictor.predict(text)
    print(f"Input text: {text}")
    print(f"Predicted act: {result['act']['label']} (confidence: {result['act']['confidence']:.2f})")
    print(f"Predicted emotion: {result['emotion']['label']} (confidence: {result['emotion']['confidence']:.2f})") 