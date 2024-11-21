"""
Modified for dialog processing with BERT tokenized input
"""
import math
import time
import torch
from torch import nn, optim
from torch.optim import Adam
import numpy as np
import os
import re

from models.model.transformer import Transformer
from models.model.sparse_autoencoder import SparseAutoencoder
from util.epoch_timer import epoch_time
from util.data_loader import DataLoader
from transformers import AutoTokenizer

def get_tokenizer(model_name='bert-base-uncased'):
    return AutoTokenizer.from_pretrained(model_name)

class DialogTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, drop_prob, device, num_act_classes, num_emotion_classes, vocab_size):
        super(DialogTransformer, self).__init__()
        self.device = device
        
        # Change Linear to Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=n_heads, 
            num_encoder_layers=n_layers, 
            num_decoder_layers=n_layers,
            dropout=drop_prob,
            batch_first=True
        )
        self.act_classifier = nn.Linear(d_model, num_act_classes)
        self.emotion_classifier = nn.Linear(d_model, num_emotion_classes)
        self.sparse_autoencoder = SparseAutoencoder(input_size=d_model, hidden_size=d_model//2)
        
    def forward(self, input_ids, attention_mask):
        # Debugging: Print shapes and types
        print(f"Input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}")
        print(f"Attention Mask shape: {attention_mask.shape}, dtype: {attention_mask.dtype}")
        
        # Embed the input_ids
        src = self.embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # Create target sequence
        tgt = torch.zeros_like(input_ids).to(self.device)  # [batch_size, seq_len]
        tgt = self.embedding(tgt)  # [batch_size, seq_len, d_model]
        
        # Create masks for transformer
        src_key_padding_mask = (~attention_mask).to(torch.bool)
        tgt_key_padding_mask = src_key_padding_mask.clone()
        
        # Create square subsequent mask for target
        tgt_mask = self.transformer.generate_square_subsequent_mask(src.size(1)).to(self.device)
        
        try:
            transformer_output = self.transformer(
                src=src,
                tgt=tgt,
                src_mask=None,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
            # Rest of the processing...
            batch_size, seq_length, feature_dim = transformer_output.size()
            transformer_output_2d = transformer_output.view(-1, feature_dim)
            
            reconstructed, kl_div, encoded = self.sparse_autoencoder(transformer_output_2d)
            reconstructed = reconstructed.view(batch_size, seq_length, feature_dim)
            
            act_output = self.act_classifier(transformer_output)
            emotion_output = self.emotion_classifier(transformer_output)
            
            return act_output, emotion_output, kl_div, reconstructed
            
        except Exception as e:
            print(f"Error during transformer processing: {e}")
            print(f"src shape: {src.shape}")
            print(f"tgt shape: {tgt.shape}")
            raise

def main(total_epoch=10, batch_size=32, learning_rate=1e-6, n_layers=6, n_heads=8, drop_prob=0.1, save_dir='default',
         alpha=0.7, beta=0.3, gamma=0.1, delta=0.1, epsilon=0.01):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    from util.dialog_loader import get_dialog_data

    # Load and prepare the dataset
    dataset = get_dialog_data()

    # Access different splits
    train_data = dataset['train']
    valid_data = dataset['validation']
    test_data = dataset['test']

    # Get number of classes from the dataset
    num_emotion_classes = 7  # DailyDialog has 7 emotion classes (0-6)
    num_act_classes = 5      # DailyDialog has 5 dialog act classes (1-5)

    # Check data distribution (optional but helpful)
    emotion_counts, act_counts = check_data_distribution(train_data)

    # Get vocabulary size from the tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    model = DialogTransformer(
        d_model=768,  # BERT's hidden size
        n_heads=n_heads,
        n_layers=n_layers,
        drop_prob=drop_prob,
        device=device,
        num_act_classes=num_act_classes,
        num_emotion_classes=num_emotion_classes,
        vocab_size=vocab_size
    ).to(device)

    # ... rest of the main function implementation ...

if __name__ == '__main__':
    main(total_epoch=5, batch_size=64, learning_rate=1e-5, n_layers=8, n_heads=12, drop_prob=0.2, save_dir="sae",
         alpha=0.7, beta=0.7, gamma=0.2, delta=0.2, epsilon=0.015)


