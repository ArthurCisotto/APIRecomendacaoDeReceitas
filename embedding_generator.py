import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

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

class RecipeDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = torch.FloatTensor(embeddings)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]

def mean_pooling(embeddings):
    """Aplica mean pooling nos embeddings"""
    return np.mean(embeddings, axis=0) if len(embeddings.shape) > 1 else embeddings

def train_recipe_encoder(df, epochs=100, batch_size=32, learning_rate=1e-4):
    # Verifica se já existem embeddings salvos
    if os.path.exists('recipe_embeddings.npy'):
        print("Carregando embeddings existentes...")
        return None, np.load('recipe_embeddings.npy'), None

    # Inicializa SBERT
    print("Inicializando modelo SBERT...")
    sbert_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    
    # Gera embeddings SBERT
    print("Gerando embeddings SBERT...")
    texts = [f"{row['ingredients']} {row['instructions']}" for _, row in df.iterrows()]
    
    # Gera embeddings e aplica mean pooling
    base_embeddings = []
    for text in tqdm(texts, desc="Gerando embeddings"):
        emb = sbert_model.encode(text)
        emb = mean_pooling(emb)
        base_embeddings.append(emb)
    
    base_embeddings = np.array(base_embeddings)
    print(f"Shape dos embeddings base: {base_embeddings.shape}")
    
    # Cria dataset e dataloader
    dataset = RecipeDataset(base_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Inicializa modelo
    model = DenoisingAutoencoder(input_dim=768)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Configuração do treinamento
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print(f"Treinando autoencoder em {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Forward pass
            decoded, encoded = model(batch, add_noise=True)
            
            # Calcula loss
            loss = criterion(decoded, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{total_loss/len(dataloader):.4f}'})
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    print("Gerando embeddings finais...")
    model.eval()
    final_embeddings = []
    
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            batch = batch.to(device)
            _, encoded = model(batch, add_noise=False)
            final_embeddings.extend(encoded.cpu().numpy())
    
    final_embeddings = np.array(final_embeddings)
    
    # Salva resultados
    torch.save(model.state_dict(), 'recipe_encoder.pt')
    np.save('recipe_embeddings.npy', final_embeddings)
    
    return model, final_embeddings, sbert_model

if __name__ == "__main__":
    # Carrega dataset
    print("Carregando dataset...")
    df = pd.read_csv('recipes.csv')
    
    # Treina modelo e gera embeddings
    model, embeddings, sbert_model = train_recipe_encoder(df)
    
    if embeddings is not None:
        print(f"Shape dos embeddings finais: {embeddings.shape}")