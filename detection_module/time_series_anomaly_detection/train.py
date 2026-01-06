import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
import os
from pathlib import Path
import json
from datetime import datetime
from LSTMAD import LSTMModel
from dataset import create_dataloader, split_csv_files
import time
from torch.utils.data import DataLoader, TensorDataset

def prepare_data(
    data: np.ndarray,
    seq_length: int,
    batch_size: int
) -> DataLoader:
    """
    Prepare data for training
    
    Args:
        data (np.ndarray): Input time series data
        seq_length (int): Length of input sequences
        batch_size (int): Batch size
        
    Returns:
        DataLoader: Data loader for training
    """
    # Create sequences using numpy array operations instead of list
    n_samples = len(data) - seq_length + 1
    sequences = np.zeros((n_samples, seq_length, data.shape[1]))
    
    for i in range(n_samples):
        sequences[i] = data[i:i + seq_length]
    
    # Convert to tensor (now from a single numpy array)
    sequences = torch.FloatTensor(sequences)
    
    # Create dataset and dataloader
    dataset = TensorDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    output_dir: Path
) -> Tuple[List[float], List[float]]:
    """
    Train the LSTM model
    
    Args:
        model: The LSTM model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        output_dir: Directory to save model and logs
        
    Returns:
        Tuple[List[float], List[float]]: Training and validation losses
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training configuration
    config = {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': train_loader.batch_size,
        'model_params': {
            'window_size': model.window_size,
            'input_size': model.input_size,
            'hidden_dim': model.hidden_dim,
            'pred_len': model.pred_len,
            'num_layers': model.num_layers
        }
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    best_val_loss = float('inf')
    total_steps = 0
    running_loss = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for idx, (x, y) in enumerate(train_loader):
            # total_steps += 1
            # print(total_steps)
            # continue
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            pred = model(x)
            
            # Calculate loss
            loss = criterion(pred, y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            running_loss += loss.item()
            total_steps += 1

            # Print loss info every 2000 steps
            if total_steps % 2000 == 0:
                avg_loss = running_loss / 2000
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{total_steps}], Average Loss: {avg_loss:.4f}')
                running_loss = 0.0

            # Save model every 50000 steps
            if total_steps % 10000 == 0 or total_steps == 1:
                torch.save({
                    'step': total_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss / (idx + 1),
                }, output_dir / f'checkpoint_step_{total_steps}.pth')

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'step': total_steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, output_dir / 'best_model.pth')
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Save final model
    torch.save({
        'step': total_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, output_dir / 'final_model.pth')
    
    # Save loss history
    np.save(output_dir / 'train_losses.npy', np.array(train_losses))
    np.save(output_dir / 'val_losses.npy', np.array(val_losses))
    
    return train_losses, val_losses

def detect_anomalies(
    model,
    data: np.ndarray,
    threshold: float,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies in the data
    
    Args:
        model (LSTMAD): Trained LSTMAD model
        data (np.ndarray): Input time series data
        threshold (float): Anomaly threshold
        device (torch.device): Device to run inference on
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Anomaly scores and binary anomaly labels
    """
    model.eval()
    with torch.no_grad():
        # Convert data to tensor
        data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
        
        # Get predictions
        _, anomaly_scores = model.predict(data_tensor)
        
        # Convert to numpy
        scores = anomaly_scores.cpu().numpy().flatten()
        
        # Get binary labels
        labels = (scores > threshold).astype(int)
        
    return scores, labels

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    window_size = 100
    input_size = 47
    hidden_dim = 64
    pred_len = 10
    num_layers = 2
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.001
    
    # Data paths
    data_dir = "path_to_your_root/results/analyze_soft/TSAD_DATASET"
    output_base = "path_to_your_root/results/analyze_soft/model_outputs"
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / f"lstm_run_{timestamp}"
    print(f"Starting EXP of lstm_run_{timestamp}")
    
    # Split CSV files into train and validation sets
    train_files, val_files = split_csv_files(data_dir, train_ratio=0.7)
    
    # Create model
    model = LSTMModel(
        window_size=window_size,
        input_size=input_size,
        hidden_dim=hidden_dim,
        pred_len=pred_len,
        num_layers=num_layers,
        batch_size=batch_size
    ).to(device)
    
    # Create dataloaders
    print(f"Creating dataloaders with {len(train_files)} training files")
    train_loader = create_dataloader(
        csv_files=train_files,
        window_size=window_size,
        pred_len=pred_len,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    print(f"Creating dataloaders with {len(val_files)} validation files")
    val_loader = create_dataloader(
        csv_files=val_files,
        window_size=window_size,
        pred_len=pred_len,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    # Train model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        output_dir=output_dir
    )
    
    print(f"Training completed. Check the output directory for results.")

    # # Detect anomalies
    # threshold = 0.1  # Adjust based on your data
    # scores, labels = detect_anomalies(model, data, threshold, device) 