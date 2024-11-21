import torch
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
import torch.nn as nn
from models.model.transformer import Transformer
from models.model.sparse_autoencoder import SparseAutoencoder

# Step 1: Define a Custom Model Class
class CustomDialogTransformer(PreTrainedModel):
    def __init__(self, config, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, device, num_act_classes, num_emotion_classes):
        super().__init__(config)
        self.transformer = Transformer(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            trg_sos_idx=trg_sos_idx,
            enc_voc_size=enc_voc_size,
            dec_voc_size=dec_voc_size,
            d_model=config.hidden_size,
            n_head=config.num_attention_heads,
            max_len=config.max_position_embeddings,
            ffn_hidden=config.intermediate_size,
            n_layers=config.num_hidden_layers,
            drop_prob=config.hidden_dropout_prob,
            device=device
        )
        
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.act_classifier = nn.Linear(config.hidden_size, num_act_classes)
        self.emotion_classifier = nn.Linear(config.hidden_size, num_emotion_classes)
        self.sparse_autoencoder = SparseAutoencoder(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            sparsity_param=0.05,
            beta=3
        )
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        transformer_output = self.transformer.encoder(input_ids, attention_mask)
        transformer_output = self.batch_norm(transformer_output.view(-1, transformer_output.size(-1)))
        transformer_output = self.dropout(transformer_output)
        reconstructed, kl_div, encoded = self.sparse_autoencoder(transformer_output)
        cls_output = reconstructed[:, 0, :]
        act_output = self.act_classifier(cls_output)
        emotion_output = self.emotion_classifier(cls_output)
        return act_output, emotion_output, kl_div

# Step 2: Load and Save the Model Weights
config = PretrainedConfig(
    hidden_size=768,  # Example configuration
    num_labels=2,     # Example number of labels
    num_attention_heads=12,
    max_position_embeddings=512,
    intermediate_size=3072,
    num_hidden_layers=12,
    hidden_dropout_prob=0.1
)

# Initialize your model with appropriate indices and sizes
model = CustomDialogTransformer(
    config=config,
    src_pad_idx=0,
    trg_pad_idx=0,
    trg_sos_idx=1,
    enc_voc_size=30522,
    dec_voc_size=30522,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    num_act_classes=5,  # Example number of act classes
    num_emotion_classes=7  # Example number of emotion classes
)

# Load the checkpoint
checkpoint = torch.load("saved/try2/model-3693.032.pt")

# Extract the model state dict
model_state_dict = checkpoint.get("model_state_dict", checkpoint)

# Filter out unexpected keys
filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in model.state_dict()}

# Load the state dict into the model
model.load_state_dict(filtered_state_dict, strict=False)

# Save the model
model.save_pretrained("path_to_model_directory")

# Step 3: Save the Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained("path_to_model_directory")

# Note: Use the Hugging Face CLI to log in and push your model
# huggingface-cli login
# transformers-cli upload path_to_model_directory