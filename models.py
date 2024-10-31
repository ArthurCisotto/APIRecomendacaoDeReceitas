import torch
import torch.nn as nn
from torch.utils.data import Dataset

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[512, 256, 128]):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        hidden_dims.reverse()
        current_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:] + [input_dim]:
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
            
        self.decoder = nn.Sequential(*decoder_layers)
        
    def add_noise(self, x, noise_factor=0.2):
        noise = torch.randn_like(x) * noise_factor
        noisy_x = x + noise
        return noisy_x
        
    def forward(self, x, add_noise=True):
        if add_noise:
            x = self.add_noise(x)
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