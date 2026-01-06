from typing import Dict
import torchinfo
import tqdm
# from ...DataFactory import TSData
# from ...Exptools import EarlyStoppingTorch
# from .. import BaseMethod
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# from .TSDataset import UTSAllInOneDataset, UTSOneByOneDataset

SOS_token = 0

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
    
# class LSTMADalpha(BaseMethod):
#     def __init__(self, params:dict) -> None:
#         super().__init__()
#         self.__anomaly_score = None
#
#         cuda = True
#         self.y_hats = None
#
#         self.cuda = cuda
#         if self.cuda == True and torch.cuda.is_available():
#             self.device = torch.device("cuda")
#             print("=== Using CUDA ===")
#         else:
#             if self.cuda == True and not torch.cuda.is_available():
#                 print("=== CUDA is unavailable ===")
#             self.device = torch.device("cpu")
#             print("=== Using CPU ===")
#
#         self.window_size = params["window_size"]
#         self.pred_len = params["pred_len"]
#         self.batch_size = params["batch_size"]
#         self.epochs = params["epochs"]
#
#         input_size = params["input_size"]
#         hidden_dim = params["hidden_dim"]
#         num_layer = params["num_layer"]
#         lr = params["lr"]
#
#         self.model = LSTMModel(self.window_size, input_size, hidden_dim, self.pred_len, num_layer, batch_size=self.batch_size, device=self.device).to(self.device)
#
#         self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
#         self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
#         self.loss = nn.MSELoss()
#         self.save_path = None
#         self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)
#
#         self.mu = None
#         self.sigma = None
#         self.eps = 1e-10
#
#
#     def anomaly_score(self) -> np.ndarray:
#         return self.__anomaly_score
#
#     def get_y_hat(self) -> np.ndarray:
#         return self.y_hats
#
#     def param_statistic(self, save_file):
#         model_stats = torchinfo.summary(self.model, (self.batch_size, self.window_size), verbose=0)
#         with open(save_file, 'w') as f:
#             f.write(str(model_stats))