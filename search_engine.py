import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import logging
import os
import pickle
from typing import List, Dict
from models import RecipeAutoencoder, RecipeDataset

logger = logging.getLogger(__name__)

class RecipeSearchEngine:
    def __init__(self, model_name='distiluse-base-multilingual-cased-v2'):
        self.sbert = SentenceTransformer(model_name)
        self.autoencoder = None
        self.df = None
        self.embeddings = None
        self.reduced_embeddings = None
        
        # Caminhos para os arquivos salvos
        self.model_dir = "saved_models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.autoencoder_path = os.path.join(self.model_dir, "autoencoder.pt")
        self.embeddings_path = os.path.join(self.model_dir, "embeddings.pkl")
        self.reduced_embeddings_path = os.path.join(self.model_dir, "reduced_embeddings.pkl")
        
    def train_autoencoder(self, embeddings, epochs=50, batch_size=32):
        dataset = RecipeDataset(embeddings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.autoencoder = RecipeAutoencoder(input_dim=embeddings.shape[1])
        optimizer = optim.Adam(self.autoencoder.parameters())
        criterion = nn.MSELoss()
        
        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                decoded, _ = self.autoencoder(batch)
                loss = criterion(decoded, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
                
        # Salvar o modelo treinado
        torch.save(self.autoencoder.state_dict(), self.autoencoder_path)
    
    def save_data(self):
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        with open(self.reduced_embeddings_path, 'wb') as f:
            pickle.dump(self.reduced_embeddings, f)
    
    def load_data(self):
        if os.path.exists(self.embeddings_path) and \
           os.path.exists(self.reduced_embeddings_path) and \
           os.path.exists(self.autoencoder_path):
            logger.info("Carregando dados e modelo salvos...")
            
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            with open(self.reduced_embeddings_path, 'rb') as f:
                self.reduced_embeddings = pickle.load(f)
                
            self.autoencoder = RecipeAutoencoder(input_dim=self.embeddings.shape[1])
            self.autoencoder.load_state_dict(torch.load(self.autoencoder_path))
            self.autoencoder.eval()
            
            return True
        return False
    
    def prepare_data(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
        # Tentar carregar dados salvos
        if self.load_data():
            return
        
        logger.info("Gerando embeddings SBERT...")
        texts = [f"{row['ingredients']} {row['instructions']}" for _, row in self.df.iterrows()]
        self.embeddings = self.sbert.encode(texts, show_progress_bar=True)
        
        logger.info("Treinando autoencoder...")
        self.train_autoencoder(self.embeddings)
        
        self.autoencoder.eval()
        with torch.no_grad():
            _, self.reduced_embeddings = self.autoencoder(torch.FloatTensor(self.embeddings))
        self.reduced_embeddings = self.reduced_embeddings.numpy()
        
        # Salvar dados processados
        self.save_data()
        logger.info("Dados preparados e salvos com sucesso")
    
    def visualize_embeddings(self, filename_prefix):
        logger.info("Gerando visualizações t-SNE...")
        
        tsne_original = TSNE(n_components=2, random_state=42)
        tsne_result_original = tsne_original.fit_transform(self.embeddings)
        
        fig_original = px.scatter(
            x=tsne_result_original[:, 0],
            y=tsne_result_original[:, 1],
            hover_data=[self.df['name']],
            title='t-SNE visualization of SBERT embeddings'
        )
        fig_original.write_html(f"{filename_prefix}_sbert.html")
        
        tsne_reduced = TSNE(n_components=2, random_state=42)
        tsne_result_reduced = tsne_reduced.fit_transform(self.reduced_embeddings)
        
        fig_reduced = px.scatter(
            x=tsne_result_reduced[:, 0],
            y=tsne_result_reduced[:, 1],
            hover_data=[self.df['name']],
            title='t-SNE visualization of autoencoder-reduced embeddings'
        )
        fig_reduced.write_html(f"{filename_prefix}_reduced.html")
    
    def search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        query_embedding = self.sbert.encode([query_text])
        
        with torch.no_grad():
            _, reduced_query = self.autoencoder(torch.FloatTensor(query_embedding))
        reduced_query = reduced_query.numpy()
        
        similarities = np.dot(self.reduced_embeddings, reduced_query.T).flatten()
        similarities = similarities / (np.linalg.norm(self.reduced_embeddings, axis=1) * np.linalg.norm(reduced_query))
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'title': self.df.iloc[idx]['name'],
                'recipe_url': self.df.iloc[idx]['url'],
                'ingredients': self.df.iloc[idx]['ingredients'],
                'relevance': float(similarities[idx])
            })
        
        return results