"""
Modified for dialog processing with BERT tokenized input
"""
import math
import time
import torch
from torch import nn, optim
from torch.optim import Adam
import numpy as np

from models.model.transformer import Transformer
from models.model.sparse_autoencoder import SparseAutoencoder
from util.epoch_timer import epoch_time
from util.data_loader import DataLoader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

class DialogTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, drop_prob, device, num_act_classes, num_emotion_classes):
        super().__init__()
        self.transformer = Transformer(
            src_pad_idx=0,  # BERT's padding token id
            trg_pad_idx=0,  # Not used but required
            trg_sos_idx=101,  # BERT's [CLS] token id, not used but required
            enc_voc_size=30522,  # BERT's vocabulary size
            dec_voc_size=30522,  # Not used but required
            d_model=d_model,
            max_len=128,    # From your tokenizer max_length
            ffn_hidden=d_model * 4,
            n_head=n_heads,
            n_layers=n_layers,
            drop_prob=drop_prob,
            device=device
        )
        
        # Add classification heads with passed-in class counts
        self.act_classifier = nn.Linear(d_model, num_act_classes)
        self.emotion_classifier = nn.Linear(d_model, num_emotion_classes)
        
        # Add SAE
        self.sparse_autoencoder = SparseAutoencoder(
            input_size=d_model,
            hidden_size=d_model//2,  # compress to half size
            sparsity_param=0.05,
            beta=3
        )

    def forward(self, input_ids, attention_mask):
        # Convert BERT-style attention mask (batch_size, seq_len) to transformer mask format (batch_size, 1, 1, seq_len)
        transformer_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Get transformer output
        transformer_output = self.transformer.encoder(input_ids, transformer_mask)
        
        # Apply SAE
        reconstructed, kl_div, encoded = self.sparse_autoencoder(transformer_output)
        
        # Use encoded representations for classification
        cls_output = reconstructed[:, 0, :]
        
        # Get predictions
        act_output = self.act_classifier(cls_output)
        emotion_output = self.emotion_classifier(cls_output)
        
        return act_output, emotion_output, kl_div

def train(model, iterator, optimizer, criterion, device, clip=1.0):
    model.train()
    epoch_loss = 0
    
    total_steps = len(iterator)
    print_interval = max(1, total_steps // 10)  # Print roughly 10 updates per epoch
    
    for i, batch in enumerate(iterator):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        acts = batch['act'].to(device)
        emotions = batch['emotion'].to(device)

        optimizer.zero_grad()
        
        # Forward pass with SAE
        act_output, emotion_output, kl_div = model(input_ids, attention_mask)
        
        # Calculate loss
        act_loss = criterion(act_output, acts)
        emotion_loss = criterion(emotion_output, emotions)
        total_loss = act_loss + emotion_loss + model.sparse_autoencoder.beta * kl_div
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += total_loss.item()
        
        # Only print at intervals
        if i % print_interval == 0 or i == total_steps - 1:
            print(f'Training step: {i}/{total_steps} ({(i/total_steps)*100:.1f}%), Loss: {total_loss.item():.4f}')

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    
    act_correct = 0
    emotion_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            acts = batch['act'].to(device)
            emotions = batch['emotion'].to(device)

            # Forward pass
            act_output, emotion_output, kl_div = model(input_ids, attention_mask)
            
            # Calculate loss
            act_loss = criterion(act_output, acts)
            emotion_loss = criterion(emotion_output, emotions)
            if kl_div.dim() == 0:  # if scalar
                kl_div = kl_div.unsqueeze(0)  # add dimension
            
            # Ensure target has same shape as kl_div
            target = torch.zeros_like(kl_div).to(device)
            kl_div_loss = criterion(kl_div, target)
            loss = act_loss + emotion_loss + kl_div_loss
            
            epoch_loss += loss.item()
            
            # Calculate accuracy
            act_pred = act_output.argmax(1)
            emotion_pred = emotion_output.argmax(1)
            
            act_correct += (act_pred == acts).sum().item()
            emotion_correct += (emotion_pred == emotions).sum().item()
            total += acts.size(0)

    act_accuracy = act_correct / total
    emotion_accuracy = emotion_correct / total
    
    return epoch_loss / len(iterator), act_accuracy, emotion_accuracy

def run(total_epoch, best_loss, model, train_iter, valid_iter, optimizer, criterion, device, scheduler, warmup=0):
    train_losses, valid_losses = [], []
    act_accuracies, emotion_accuracies = [], []
    best_checkpoint_path = None
    
    for step in range(total_epoch):
        start_time = time.time()
        
        train_loss = train(model, train_iter, optimizer, criterion, device, clip=1.0)
        valid_loss, act_acc, emotion_acc = evaluate(model, valid_iter, criterion, device)
        
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            checkpoint_path = f'saved/model-{valid_loss:.3f}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            best_checkpoint_path = checkpoint_path  # Track the best checkpoint

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tValid Loss: {valid_loss:.3f}')
        print(f'\tAct Accuracy: {act_acc:.3f}')
        print(f'\tEmotion Accuracy: {emotion_acc:.3f}')
    
    # Print final notice about best checkpoint
    if best_checkpoint_path:
        print("\n" + "="*50)
        print(f"Training completed! Best checkpoint saved at:")
        print(f"{best_checkpoint_path}")
        print(f"Best validation loss: {best_loss:.3f}")
        print("="*50 + "\n")

def collate_batch(batch):
    batch_dict = {
        'input_ids': [],
        'attention_mask': [],
        'act': [],
        'emotion': []
    }
    
    for item in batch:
        for key in batch_dict:
            # For act and emotion, take only the last label in the sequence
            if key in ['act', 'emotion']:
                batch_dict[key].append(item[key][-1])  # Take the last label
            else:
                batch_dict[key].append(item[key])
    
    # Stack tensors
    batch_dict['input_ids'] = torch.stack(batch_dict['input_ids'])
    batch_dict['attention_mask'] = torch.stack(batch_dict['attention_mask'])
    
    # Convert to list of integers first
    acts = [int(act) if isinstance(act, (int, float)) else int(act.item()) for act in batch_dict['act']]
    emotions = [int(emotion) if isinstance(emotion, (int, float)) else int(emotion.item()) for emotion in batch_dict['emotion']]
    
    # Convert lists to tensors
    batch_dict['act'] = torch.tensor(acts, dtype=torch.long)
    batch_dict['emotion'] = torch.tensor(emotions, dtype=torch.long)
    
    return batch_dict

def main(total_epoch=1, batch_size=32):
    from util.dialog_loader import get_dialog_data

    # Load and prepare the dataset
    dataset = get_dialog_data()

    # Access different splits
    train_data = dataset['train']
    valid_data = dataset['validation']
    test_data = dataset['test']

    # Create DataLoader objects
    train_iter = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
    
    valid_iter = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    # Hyperparameters
    d_model = 768  # BERT's hidden size
    n_heads = 8
    n_layers = 6
    drop_prob = 0.1
    num_act_classes = len(set(train_data['act']))  # Get from your data
    num_emotion_classes = len(set(train_data['emotion']))  # Get from your data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Initialize model with class counts
    model = DialogTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        drop_prob=drop_prob,
        device=device,
        num_act_classes=num_act_classes,
        num_emotion_classes=num_emotion_classes
    ).to(device)
    
    # Initialize optimizer and criterion
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    # Run training
    run(
        total_epoch=total_epoch,
        best_loss=float('inf'),
        model=model,
        train_iter=train_iter,
        valid_iter=valid_iter,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        warmup=0
    )
    
    # Collect SAE metadata
    print("Collecting SAE metadata...")
    model.eval()
    sae_metadata = {}
    with torch.no_grad():
        activations = []
        
        for i, batch in enumerate(valid_iter):
            if i >= 10:  # Collect data from 10 batches
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get model outputs directly
            _, _, encoded = model(input_ids, attention_mask)
            
            # Debug print
            print(f"Debug - encoded shape before processing: {encoded.shape}")
            
            # Handle scalar outputs
            if encoded.dim() == 0:  # scalar
                encoded = encoded.unsqueeze(0).unsqueeze(0)  # Add batch and feature dimensions
            elif encoded.dim() == 1:  # vector
                encoded = encoded.unsqueeze(0)  # Add batch dimension
                
            print(f"Debug - encoded shape after processing: {encoded.shape}")
            
            # Convert to numpy and store
            encoded_np = encoded.cpu().numpy()
            activations.append(encoded_np)
    
    # Concatenate and store metadata
    if len(activations) > 0:
        try:
            # Ensure all arrays have the same shape
            shapes = [arr.shape for arr in activations]
            print(f"Shapes of collected activations: {shapes}")
            
            # Reshape if needed to ensure 2D arrays [batch, features]
            processed_activations = []
            for arr in activations:
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                elif arr.ndim > 2:
                    arr = arr.reshape(-1, arr.shape[-1])
                processed_activations.append(arr)
            
            sae_metadata['activations'] = np.concatenate(processed_activations, axis=0)
            print(f"Successfully collected activations with shape: {sae_metadata['activations'].shape}")
        except Exception as e:
            print(f"Error concatenating activations: {e}")
            print(f"Shapes of activations: {[a.shape for a in activations]}")
    else:
        print("Warning: No activations collected!")
    
    # Print final metadata keys
    print(f"Metadata keys: {sae_metadata.keys()}")
    
    return model, valid_iter, device, sae_metadata

if __name__ == '__main__':
    main(total_epoch=3, batch_size=32)
