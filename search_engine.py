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
import re
from typing import List, Dict
from tqdm import tqdm
from models import DenoisingAutoencoder, RecipeDataset

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
    
    def mean_pooling(self, embeddings):
        """Aplica mean pooling nos embeddings"""
        return np.mean(embeddings, axis=0) if len(embeddings.shape) > 1 else embeddings
        
    def train_autoencoder(self, embeddings, epochs=100, batch_size=32):
        dataset = RecipeDataset(embeddings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.autoencoder = DenoisingAutoencoder(
            input_dim=embeddings.shape[1],
            hidden_dims=[512, 256, 128]
        )
        
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                decoded, _ = self.autoencoder(batch, add_noise=True)
                loss = criterion(decoded, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
                
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
                
            self.autoencoder = DenoisingAutoencoder(input_dim=self.embeddings.shape[1])
            self.autoencoder.load_state_dict(torch.load(self.autoencoder_path))
            self.autoencoder.eval()
            
            return True
        return False
    
    def prepare_data(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
        if self.load_data():
            return
        
        logger.info("Gerando embeddings SBERT...")
        texts = [f"{row['ingredients']} {row['instructions']}" for _, row in self.df.iterrows()]
        
        # Gera embeddings e aplica mean pooling
        embeddings = []
        for text in tqdm(texts, desc="Gerando embeddings"):
            emb = self.sbert.encode(text)
            emb = self.mean_pooling(emb)
            embeddings.append(emb)
        
        self.embeddings = np.array(embeddings)
        
        logger.info("Treinando denoising autoencoder...")
        self.train_autoencoder(self.embeddings)
        
        self.autoencoder.eval()
        with torch.no_grad():
            _, self.reduced_embeddings = self.autoencoder(
                torch.FloatTensor(self.embeddings),
                add_noise=False
            )
        self.reduced_embeddings = self.reduced_embeddings.numpy()
        
        self.save_data()
        logger.info("Dados preparados e salvos com sucesso")
    
    def visualize_embeddings(self, filename_prefix):
        sbert_viz_path = f"{filename_prefix}_sbert.html"
        reduced_viz_path = f"{filename_prefix}_reduced.html"
        
        if os.path.exists(sbert_viz_path) and os.path.exists(reduced_viz_path):
            logger.info("Visualizações já existem, pulando geração...")
            return
            
        logger.info("Gerando visualizações t-SNE...")
        
        tsne_original = TSNE(n_components=2, random_state=42)
        tsne_result_original = tsne_original.fit_transform(self.embeddings)
        
        fig_original = px.scatter(
            x=tsne_result_original[:, 0],
            y=tsne_result_original[:, 1],
            hover_data=[self.df['name']],
            title='t-SNE visualization of SBERT embeddings'
        )
        fig_original.write_html(sbert_viz_path)
        
        tsne_reduced = TSNE(n_components=2, random_state=42)
        tsne_result_reduced = tsne_reduced.fit_transform(self.reduced_embeddings)
        
        fig_reduced = px.scatter(
            x=tsne_result_reduced[:, 0],
            y=tsne_result_reduced[:, 1],
            hover_data=[self.df['name']],
            title='t-SNE visualization of autoencoder-reduced embeddings'
        )
        fig_reduced.write_html(reduced_viz_path)
        
        logger.info("Visualizações geradas com sucesso")

    def normalize_text(self, text: str) -> str:
        """Normaliza o texto removendo acentos e convertendo para minúsculas"""
        text = text.lower()
        text = re.sub(r'[áàãâä]', 'a', text)
        text = re.sub(r'[éèêë]', 'e', text)
        text = re.sub(r'[íìîï]', 'i', text)
        text = re.sub(r'[óòõôö]', 'o', text)
        text = re.sub(r'[úùûü]', 'u', text)
        text = re.sub(r'[ç]', 'c', text)
        return text
    
    def ingredient_similarity_score(self, query: str, ingredients: str) -> float:
        """Calcula um score baseado na presença explícita dos ingredientes"""
        query = self.normalize_text(query)
        ingredients = self.normalize_text(ingredients)
        
        if query in ingredients:
            return 1.0
            
        query_words = set(query.split())
        ingredient_words = set(ingredients.split())
        common_words = query_words.intersection(ingredient_words)
        
        if not common_words:
            return 0.0
            
        return len(common_words) / len(query_words)
    
    def search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        query_embedding = self.sbert.encode([query_text])
        
        with torch.no_grad():
            _, reduced_query = self.autoencoder(torch.FloatTensor(query_embedding), add_noise=False)
        reduced_query = reduced_query.numpy()
        
        embedding_similarities = np.dot(self.reduced_embeddings, reduced_query.T).flatten()
        embedding_similarities = embedding_similarities / (np.linalg.norm(self.reduced_embeddings, axis=1) * np.linalg.norm(reduced_query))
        
        ingredient_scores = np.array([
            self.ingredient_similarity_score(query_text, ingredients)
            for ingredients in self.df['ingredients']
        ])
        
        combined_scores = (0.7 * embedding_similarities) + (0.3 * ingredient_scores)
        valid_indices = ingredient_scores > 0
        
        if valid_indices.any():
            filtered_scores = combined_scores[valid_indices]
            filtered_indices = np.arange(len(combined_scores))[valid_indices]
            top_k_filtered = min(top_k, len(filtered_scores))
            top_indices = filtered_indices[filtered_scores.argsort()[-top_k_filtered:][::-1]]
        else:
            top_indices = combined_scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0.5:
                results.append({
                    'title': self.df.iloc[idx]['name'],
                    'recipe_url': self.df.iloc[idx]['url'],
                    'ingredients': self.df.iloc[idx]['ingredients'],
                    'relevance': float(combined_scores[idx]),
                    'ingredient_match': float(ingredient_scores[idx])
                })
        
        return results