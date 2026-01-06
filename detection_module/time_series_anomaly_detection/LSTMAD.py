import torch
import torch.nn as nn
from typing import Tuple
import time


class LSTMModel(nn.Module):
    def __init__(self, window_size, input_size,
                 hidden_dim, pred_len, num_layers, batch_size) -> None:
        """
        LSTM Model for time series prediction
        
        Args:
            window_size (int): Length of input sequence
            input_size (int): Number of input features
            hidden_dim (int): Number of hidden units in LSTM
            pred_len (int): Length of prediction sequence
            num_layers (int): Number of LSTM layers
            batch_size (int): Batch size
        """
        super().__init__()
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.window_size = window_size

        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # LSTM decoder
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Output projection layer
        self.fc = nn.Linear(hidden_dim, input_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, window_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, pred_len, input_size)
        """
        # Start timing

        encoder_outputs, (hidden, cell) = self.lstm_encoder(src)
        
        # Prepare decoder input (use last encoder output as initial input)
        decoder_input = encoder_outputs[:, -1:, :]  # (batch_size, 1, hidden_dim)
        
        # Initialize output tensor
        outputs = []
        
        # Decoder
        for _ in range(self.pred_len):
            # Decode one step
            decoder_output, (hidden, cell) = self.lstm_decoder(decoder_input, (hidden, cell))
            
            # Project to output dimension
            output = self.fc(decoder_output)  # (batch_size, 1, input_size)
            outputs.append(output)
            
            # Use current output as next input
            decoder_input = decoder_output
        
        # Concatenate all outputs
        outputs = torch.cat(outputs, dim=1)  # (batch_size, pred_len, input_size)
        
        return outputs
