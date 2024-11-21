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

import sys
import os
from contextlib import contextmanager

@contextmanager
def log_to_file(log_file_path):
    """Context manager to redirect stdout to a log file."""
    original_stdout = sys.stdout
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
        try:
            yield
        finally:
            sys.stdout = original_stdout

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
        
        # Correct the feature size for batch normalization
        self.batch_norm = nn.BatchNorm1d(d_model)  # Ensure d_model matches the transformer output feature size
        
        # Adjust dropout rate
        self.dropout = nn.Dropout(p=drop_prob)
        
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
        
        # Reshape transformer output for batch normalization
        batch_size, seq_len, feature_dim = transformer_output.size()
        transformer_output = transformer_output.view(batch_size * seq_len, feature_dim)
        
        # Apply batch normalization
        transformer_output = self.batch_norm(transformer_output)
        
        # Reshape back to original dimensions
        transformer_output = transformer_output.view(batch_size, seq_len, feature_dim)
        
        # Apply dropout
        transformer_output = self.dropout(transformer_output)
        
        # Apply SAE
        reconstructed, kl_div, encoded = self.sparse_autoencoder(transformer_output)
        
        # Use encoded representations for classification
        cls_output = reconstructed[:, 0, :]
        
        # Get predictions
        act_output = self.act_classifier(cls_output)
        emotion_output = self.emotion_classifier(cls_output)
        
        return act_output, emotion_output, kl_div

def get_class_weights(counts, num_classes):
    """Calculate class weights based on inverse frequency"""
    total = sum(counts.values())
    weights = torch.zeros(num_classes)
    for cls, count in counts.items():
        weights[cls] = total / (count * num_classes)
    return weights

def balanced_accuracy(preds, labels, num_classes):
    """Calculate balanced accuracy per class"""
    accuracies = []
    for cls in range(num_classes):
        mask = (labels == cls)
        if mask.sum() > 0:  # if we have samples for this class
            class_acc = ((preds == labels) & mask).float().sum() / mask.float().sum()
            accuracies.append(class_acc)
    return torch.tensor(accuracies).mean()

def focal_loss(logits, targets, alpha=1, gamma=2):
    """Compute focal loss for classification tasks."""
    ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

def calculate_total_loss(emotion_output, emotions, act_output, acts, kl_div, emotion_acc, act_acc, alpha=0.7, beta=0.3, gamma=0.1, delta=0.1, epsilon=0.001):
    # Calculate emotion and act losses using focal loss
    emotion_loss = focal_loss(emotion_output, emotions, alpha=0.25, gamma=2.0)
    act_loss = focal_loss(act_output, acts, alpha=0.25, gamma=2.0)
    
    # Calculate accuracy penalties
    emotion_acc_penalty = (1 - emotion_acc)
    act_acc_penalty = (1 - act_acc)
    
    # Calculate total loss
    total_loss = (alpha * emotion_loss + beta * act_loss +
                  gamma * emotion_acc_penalty + delta * act_acc_penalty +
                  epsilon * kl_div)
    
    # Debugging: Print individual loss components
    print(f"Emotion Loss: {emotion_loss.item()}, Act Loss: {act_loss.item()}, KL Div: {kl_div.item()}")
    print(f"Emotion Acc Penalty: {emotion_acc_penalty.item()}, Act Acc Penalty: {act_acc_penalty.item()}")
    print(f"Total Loss: {total_loss.item()}")
    
    return total_loss

def train(model, iterator, optimizer, criterion_emotion, criterion_act, device, clip=1.0, alpha=0.7, beta=0.3, gamma=0.1, delta=0.1, epsilon=0.001):
    model.train()
    epoch_loss = 0
    
    total_steps = len(iterator)
    
    # Initialize per-class metrics
    num_emotion_classes = 7  # Adjust this based on your actual number of emotion classes
    num_act_classes = 5      # Adjust this based on your actual number of act classes
    emotion_correct_per_class = torch.zeros(num_emotion_classes).to(device)
    emotion_total_per_class = torch.zeros(num_emotion_classes).to(device)
    act_correct_per_class = torch.zeros(num_act_classes).to(device)
    act_total_per_class = torch.zeros(num_act_classes).to(device)
    
    # Define print interval
    print_interval = 10  # Adjust this value as needed
    
    for i, batch in enumerate(iterator):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        acts = batch['act'].to(device)
        emotions = batch['emotion'].to(device)

        optimizer.zero_grad()
        
        # Forward pass
        act_output, emotion_output, kl_div = model(input_ids, attention_mask)
        
        # Calculate accuracy
        with torch.no_grad():
            emotion_preds = emotion_output.argmax(1)
            act_preds = act_output.argmax(1)
            
            # Update per-class metrics
            for cls in range(num_emotion_classes):
                mask = (emotions == cls)
                emotion_correct_per_class[cls] += ((emotion_preds == emotions) & mask).sum()
                emotion_total_per_class[cls] += mask.sum()
            
            for cls in range(num_act_classes):
                mask = (acts == cls)
                act_correct_per_class[cls] += ((act_preds == acts) & mask).sum()
                act_total_per_class[cls] += mask.sum()

        # Calculate balanced accuracies
        emotion_acc = torch.where(
            emotion_total_per_class > 0,
            emotion_correct_per_class / emotion_total_per_class,
            torch.tensor(0.0).to(device)
        ).mean()
        
        act_acc = torch.where(
            act_total_per_class > 0,
            act_correct_per_class / act_total_per_class,
            torch.tensor(0.0).to(device)
        ).mean()
        
        # Calculate total loss using the new function
        total_loss = calculate_total_loss(
            emotion_output, emotions, act_output, acts, kl_div, emotion_acc, act_acc,
            alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon
        )
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # Accumulate loss
        epoch_loss += total_loss.item()

        # Print progress
        if i % print_interval == 0 or i == total_steps - 1:
            print(f'Step: {i}/{total_steps} ({(i/total_steps)*100:.1f}%)')
            print(f'  Total Loss: {total_loss.item():.4f}')

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion_emotion, criterion_act, device, alpha=0.7, beta=0.3, gamma=0.1, delta=0.1, epsilon=0.001):
    model.eval()
    epoch_loss = 0
    
    # Per-class metrics
    emotion_correct_per_class = torch.zeros(7).to(device)
    emotion_total_per_class = torch.zeros(7).to(device)
    act_correct_per_class = torch.zeros(5).to(device)
    act_total_per_class = torch.zeros(5).to(device)
    
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            acts = batch['act'].to(device)
            emotions = batch['emotion'].to(device)

            # Forward pass
            act_output, emotion_output, kl_div = model(input_ids, attention_mask)
            
            # Calculate accuracy
            emotion_preds = emotion_output.argmax(1)
            act_preds = act_output.argmax(1)
            
            for cls in range(7):
                mask = (emotions == cls)
                emotion_correct_per_class[cls] += ((emotion_preds == emotions) & mask).sum()
                emotion_total_per_class[cls] += mask.sum()
            
            for cls in range(5):
                mask = (acts == cls)
                act_correct_per_class[cls] += ((act_preds == acts) & mask).sum()
                act_total_per_class[cls] += mask.sum()

            # Calculate balanced accuracies
            emotion_acc = torch.where(
                emotion_total_per_class > 0,
                emotion_correct_per_class / emotion_total_per_class,
                torch.tensor(0.0).to(device)
            ).mean()
            
            act_acc = torch.where(
                act_total_per_class > 0,
                act_correct_per_class / act_total_per_class,
                torch.tensor(0.0).to(device)
            ).mean()
            
            # Calculate total loss using the new function
            total_loss = calculate_total_loss(
                emotion_output, emotions, act_output, acts, kl_div, emotion_acc, act_acc,
                alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon
            )
            
            epoch_loss += total_loss.item()

    return epoch_loss / len(iterator), emotion_acc, act_acc

def collect_sae_metadata(model, valid_iter, device):
    model.eval()
    all_encoded_features = []
    all_reconstruction_errors = []
    all_kl_divs = []
    
    with torch.no_grad():
        # Collect data from multiple batches for better statistics
        for batch_idx, batch in enumerate(valid_iter):
            if batch_idx >= 5:  # Limit to 5 batches for efficiency
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get model outputs
            hidden_states = model.transformer.encoder(input_ids, attention_mask.unsqueeze(1).unsqueeze(2))
            reconstructed, kl_div, encoded = model.sparse_autoencoder(hidden_states)
            
            # Reshape encoded features to 2D: (batch_size * sequence_length, feature_dim)
            batch_size, seq_len, feature_dim = encoded.shape
            encoded_2d = encoded.reshape(-1, feature_dim)
            
            # Collect metrics
            all_encoded_features.append(encoded_2d.cpu().numpy())
            all_reconstruction_errors.append(((reconstructed - hidden_states) ** 2).mean(dim=-1).cpu().numpy())
            
            # Reshape KL divergence if it's a scalar
            if isinstance(kl_div, torch.Tensor):
                kl_div = kl_div.cpu().numpy()
            if np.isscalar(kl_div):
                kl_div = np.array([kl_div])
            all_kl_divs.append(kl_div)
    
    # Combine all batches
    encoded_features = np.concatenate(all_encoded_features, axis=0)  # Now 2D: (total_samples, feature_dim)
    reconstruction_errors = np.concatenate(all_reconstruction_errors, axis=0).flatten()
    kl_divergences = np.array(all_kl_divs)
    
    # Format metadata to match SAEAnalyzer expectations
    sae_metadata = {
        'encoded_features': encoded_features,
        'reconstruction_errors': reconstruction_errors,
        'kl_divergences': kl_divergences,
        'feature_means': np.mean(encoded_features, axis=0),
        'feature_stds': np.std(encoded_features, axis=0),
        'dead_features': np.where(np.mean(encoded_features != 0, axis=0) < 0.01)[0],
        'activation_rates': np.mean(encoded_features != 0, axis=0),
        'correlations': np.corrcoef(encoded_features.T)  # Now works with 2D array
    }
    
    return sae_metadata

def run(total_epoch, best_loss, model, train_iter, valid_iter, optimizer, criterion_emotion, criterion_act, device, scheduler, warmup=0, save_dir='default',
         alpha=0.7, beta=0.3, gamma=0.1, delta=0.1, epsilon=0.01):
    train_losses, valid_losses = [], []
    emotion_accuracies = []
    
    # Initialize checkpoint_info with existing checkpoints
    parent_dir = os.path.join('saved', save_dir)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    # Scan for existing checkpoints
    checkpoint_info = []
    for filename in os.listdir(parent_dir):
        if filename.startswith('model-') and filename.endswith('.pt'):
            loss_str = filename[6:-3]  # Extract loss value from filename
            try:
                loss = float(loss_str)
                checkpoint_path = os.path.join(parent_dir, filename)
                sae_path = os.path.join(parent_dir, f'sae_analysis-{loss_str}.npz')
                checkpoint_info.append({
                    'path': checkpoint_path,
                    'sae_path': sae_path,
                    'loss': loss
                })
            except ValueError:
                continue
    
    print(f"\nFound {len(checkpoint_info)} existing checkpoints")
    
    # Rest of the training loop...
    for step in range(total_epoch):
        start_time = time.time()
        
        train_loss = train(model, train_iter, optimizer, criterion_emotion, criterion_act, device, clip=1.0,
                           alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon)
        
        valid_loss, emotion_acc, act_acc = evaluate(model, valid_iter, criterion_emotion, criterion_act, device,
                                                    alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon)
        
        end_time = time.time()

        if step > warmup:
            scheduler.step()
        
        parent_dir = os.path.join('saved', save_dir)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        # Collect and save SAE metadata every 5 epochs
        if step % 5 == 0:  # Save every 5 epochs to reduce overhead
            sae_metadata = collect_sae_metadata(model, valid_iter, device)
            
            # Save checkpoint with SAE data
            checkpoint_path = os.path.join(parent_dir, f'model-{valid_loss:.3f}.pt')
            sae_analysis_path = os.path.join(parent_dir, f'sae_analysis-{valid_loss:.3f}.npz')
            
            torch.save({
                'epoch': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'best_loss': best_loss,
                'sae_metadata': sae_metadata
            }, checkpoint_path)
            
            # Save SAE analysis file
            np.savez(sae_analysis_path, **sae_metadata)
            
            checkpoint_info.append({
                'path': checkpoint_path, 
                'sae_path': sae_analysis_path,
                'loss': valid_loss
            })
            
            # Sort checkpoints by loss
            checkpoint_info.sort(key=lambda x: x['loss'])
            
            # Keep only the top 2 checkpoints
            if len(checkpoint_info) > 2:
                for checkpoint in checkpoint_info[2:]:
                    try:
                        if os.path.exists(checkpoint['path']):
                            os.remove(checkpoint['path'])
                            print(f"Removed checkpoint: {checkpoint['path']}")
                        if os.path.exists(checkpoint['sae_path']):
                            os.remove(checkpoint['sae_path'])
                            print(f"Removed SAE analysis: {checkpoint['sae_path']}")
                    except Exception as e:
                        print(f"Error removing files: {e}")
                checkpoint_info = checkpoint_info[:2]

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print("================================================")
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tValid Loss: {valid_loss:.3f}')
        print(f'\tEmotion Accuracy: {emotion_acc:.3f}')
        print(f'\Acct Accuracy: {act_acc:.3f}')
        print("================================================")
    
    # Post-training cleanup
    print("\nPost-training cleanup: Keeping only the best 2 checkpoints and their SAE analyses...")
    checkpoint_info.sort(key=lambda x: x['loss'])
    checkpoints_to_delete = checkpoint_info[2:]
    
    for checkpoint in checkpoints_to_delete:
        try:
            if os.path.exists(checkpoint['path']):
                os.remove(checkpoint['path'])
                print(f"Removed checkpoint after training: {checkpoint['path']}")
            if os.path.exists(checkpoint['sae_path']):
                os.remove(checkpoint['sae_path'])
                print(f"Removed SAE analysis after training: {checkpoint['sae_path']}")
        except Exception as e:
            print(f"Error removing files: {e}")
    
    # Update checkpoint_info
    checkpoint_info = checkpoint_info[:2]
    
    # Final report
    print("\n" + "="*50)
    print(f"Training completed! Best checkpoints and SAE analyses saved at:")
    for i, checkpoint in enumerate(checkpoint_info):
        print(f"#{i+1} checkpoint: {checkpoint['path']} with loss: {checkpoint['loss']:.3f}")
        print(f"    SAE analysis: {checkpoint['sae_path']}")
    print("="*50 + "\n")

    return model, checkpoint_info, sae_metadata

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

def find_best_checkpoint(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    """Find the checkpoint with the smallest loss in the given directory."""
    checkpoint_files = [f for f in os.listdir(directory) if f.startswith('model-') and f.endswith('.pt')]
    if not checkpoint_files:
        return None
    
    # Extract loss values from filenames
    loss_pattern = re.compile(r'model-(\d+\.\d+)\.pt')
    losses = [(float(loss_pattern.search(f).group(1)), f) for f in checkpoint_files if loss_pattern.search(f)]
    
    # Find the file with the smallest loss
    _, best_checkpoint = min(losses, key=lambda x: x[0])
    return os.path.join(directory, best_checkpoint)

def load_checkpoint(model, optimizer, directory):
    checkpoint_path = find_best_checkpoint(directory)
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch} with best loss {best_loss}.")
            return model, optimizer, start_epoch, best_loss
        except KeyError as e:
            print(f"KeyError: {e}. Checkpoint file might be corrupted or have missing keys.")
        except Exception as e:
            print(f"An error occurred while loading the checkpoint: {e}")
    else:
        print("No checkpoint found, starting from scratch.")
    
    return model, optimizer, 0, float('inf')

def check_data_distribution(train_data):
    """Check the distribution of emotions and acts in the dataset"""
    emotion_counts = {}
    act_counts = {}
    
    # Get class names
    emotion_names = train_data.features['emotion'].feature.names
    act_names = train_data.features['act'].feature.names
    
    # Count emotions and acts
    for i in range(len(train_data)):
        sample = train_data[i]
        emotion_val = sample['emotion'][-1].item() if torch.is_tensor(sample['emotion'][-1]) else sample['emotion'][-1]
        act_val = sample['act'][-1].item() if torch.is_tensor(sample['act'][-1]) else sample['act'][-1]
        
        emotion_counts[emotion_val] = emotion_counts.get(emotion_val, 0) + 1
        act_counts[act_val] = act_counts.get(act_val, 0) + 1
    
    # Print summarized distribution
    print(f"\nDataset Summary:")
    print(f"Total samples: {len(train_data)}")
    print(f"Number of emotion classes: {len(emotion_names)}")
    print(f"Number of act classes: {len(act_names)}")
    
    # Find majority classes
    max_emotion = max(emotion_counts.items(), key=lambda x: x[1])
    max_act = max(act_counts.items(), key=lambda x: x[1])
    
    print(f"\nMajority emotion: {emotion_names[max_emotion[0]]} ({max_emotion[1]/len(train_data)*100:.1f}%)")
    print(f"Majority act: {act_names[max_act[0]]} ({max_act[1]/len(train_data)*100:.1f}%)")

    return emotion_counts, act_counts

def main(total_epoch=10, batch_size=32, learning_rate=1e-6, n_layers=6, n_heads=8, drop_prob=0.1, save_dir='default',
         alpha=0.7, beta=0.3, gamma=0.1, delta=0.1, epsilon=0.01):
    from util.dialog_loader import get_dialog_data

    # Load and prepare the dataset
    dataset = get_dialog_data()

    # Access different splits
    train_data = dataset['train']
    valid_data = dataset['validation']
    test_data = dataset['test']

    # Only check training data distribution once
    print("\n=== Data Distribution Analysis ===")
    train_dist = check_data_distribution(train_data)
    
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
    num_act_classes = len(train_data.features['act'].feature.names)
    num_emotion_classes = len(train_data.features['emotion'].feature.names)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = DialogTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        drop_prob=drop_prob,
        device=device,
        num_act_classes=num_act_classes,
        num_emotion_classes=num_emotion_classes
    ).to(device)
    
    # Get class weights from distribution
    emotion_weights = get_class_weights(train_dist[0], num_emotion_classes).to(device)
    act_weights = get_class_weights(train_dist[1], num_act_classes).to(device)
    
    # Create weighted loss functions
    criterion_emotion = nn.CrossEntropyLoss(weight=emotion_weights, label_smoothing=0.1)
    criterion_act = nn.CrossEntropyLoss(weight=act_weights, label_smoothing=0.1)
    
    # Initialize optimizer with a smaller learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-5,
        epochs=total_epoch,
        steps_per_epoch=len(train_iter),
        pct_start=0.1
    )
    
    # Load checkpoint if exists
    model, optimizer, start_epoch, best_loss = load_checkpoint(
        model, optimizer, os.path.join('saved', save_dir)
    )
    
    # Define the log file path
    log_file_path = os.path.join('saved', save_dir, 'training_log.txt')
    
    # Use the context manager to redirect stdout to the log file
    with log_to_file(log_file_path):
        print("=== Starting Training ===")
        
        # Run training
        model, checkpoint_info, sae_metadata = run(
            total_epoch=total_epoch,
            best_loss=best_loss,
            model=model,
            train_iter=train_iter,
            valid_iter=valid_iter,
            optimizer=optimizer,
            criterion_emotion=criterion_emotion,  # Pass emotion criterion
            criterion_act=criterion_act,         # Pass act criterion
            device=device,
            scheduler=scheduler,
            warmup=0,
            save_dir=save_dir,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            epsilon=epsilon
        )
        
        print("=== Training Completed ===")

if __name__ == '__main__':
    """
    Loss Parameters:
    - alpha: Weight for the emotion loss component.
    - beta: Weight for the act loss component.
    - gamma: Weight for the penalty based on emotion accuracy (1 - emotion accuracy).
    - delta: Weight for the penalty based on act accuracy (1 - act accuracy).
    - epsilon: Scaling factor for the KL divergence component to control its impact on the total loss.
    """


    main(total_epoch=20, batch_size=64, learning_rate=1e-5, n_layers=8, n_heads=12, drop_prob=0.2, save_dir="try_new_64",
         alpha=0.7, beta=0.7, gamma=0.2, delta=0.2, epsilon=0.015)


