import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os

class RecipeAutoencoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=384):
        super(RecipeAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def encode(self, x):
        return self.encoder(x)

class RecipeDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = torch.FloatTensor(embeddings)
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]
