import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, PreTrainedModel, AutoTokenizer

# Define a custom Hugging Face-compatible model
class CustomHFBertModel(PreTrainedModel):
    config_class = BertConfig  # Use Hugging Face's BertConfig

    def __init__(self, config, num_act_classes, num_emotion_classes):
        super().__init__(config)
        # Load Hugging Face's BertModel
        self.bert = BertModel(config)
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.act_classifier = nn.Linear(config.hidden_size, num_act_classes)
        self.emotion_classifier = nn.Linear(config.hidden_size, num_emotion_classes)

        # Initialize weights
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        # Get the BERT outputs
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        # Pool the CLS token output
        pooled_output = bert_output.pooler_output
        # Apply batch normalization and dropout
        pooled_output = self.batch_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        # Get act and emotion classification outputs
        act_output = self.act_classifier(pooled_output)
        emotion_output = self.emotion_classifier(pooled_output)
        return act_output, emotion_output


# Load the checkpoint and reconstruct the model
def reconstruct_model(checkpoint_path, save_path):
    # Step 1: Define the configuration
    config = BertConfig(
        hidden_size=768,  # Example hidden size
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
    )

    # Step 2: Initialize the model with act and emotion classification heads
    model = CustomHFBertModel(
        config=config,
        num_act_classes=5,  # Example number of act classes
        num_emotion_classes=7,  # Example number of emotion classes
    )

    # Step 3: Load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # Check if checkpoint contains "model_state_dict"
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint

    # Step 4: Filter and load the state dict
    print("Loading state dict into model...")
    filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict, strict=False)

    # Step 5: Save the reconstructed model
    print(f"Saving reconstructed model to {save_path}")
    model.save_pretrained(save_path)

    # Step 6: Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(save_path)

    print("Model and tokenizer saved successfully!")


# Paths
checkpoint_path = "saved/try2/model-3693.032.pt"  # Path to your checkpoint
save_path = "reconstructed_model"  # Directory to save the reconstructed model

# Run the reconstruction
reconstruct_model(checkpoint_path, save_path)

# Test loading the model and tokenizer
print("\nTesting the reconstructed model...")
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(save_path)

# Perform a test forward pass
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)

print("Model test successful. Outputs:")
print(outputs)
