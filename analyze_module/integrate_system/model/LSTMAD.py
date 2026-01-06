import torch
import torch.nn as nn
from typing import Tuple
import time


class LSTMModel(nn.Module):
    def __init__(self, window_size, input_size, 
                 hidden_dim, pred_len, num_layers, batch_size, device) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.input_size = input_size
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        self.lstm_encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm_decoder = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        self.relu = nn.GELU()
        self.fc = nn.Linear(hidden_dim, input_size)
        
    def forward(self, src):
        # src = torch.unsqueeze(src, -1)
        _, decoder_hidden = self.lstm_encoder(src)
        cur_batch = src.shape[0]
        
        decoder_input = torch.zeros(cur_batch, 1, self.input_size).to(self.device)
        outputs = torch.zeros(self.pred_len, cur_batch, self.input_size).to(self.device)
        
        for t in range(self.pred_len):
            decoder_output, decoder_hidden = self.lstm_decoder(decoder_input, decoder_hidden)
            decoder_output = self.relu(decoder_output)
            decoder_input = self.fc(decoder_output)
            
            outputs[t] = torch.squeeze(decoder_input, dim=-2)
        outputs = torch.transpose(outputs, 0, 1)
        return outputs
