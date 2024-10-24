import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import logging
from typing import List, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Autoencoder para refinamento dos embeddings
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

# Dataset personalizado para receitas
class RecipeDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = torch.FloatTensor(embeddings)
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]

class RecipeSearchEngine:
    def __init__(self, model_name='distiluse-base-multilingual-cased-v2'):
        self.sbert = SentenceTransformer(model_name)
        self.autoencoder = None
        self.df = None
        self.embeddings = None
        self.reduced_embeddings = None
        
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
    
    def prepare_data(self, csv_path):
        self.df = pd.read_csv(csv_path)
        logger.info("Generating SBERT embeddings...")
        
        # Combine ingredients and instructions for better semantic understanding
        texts = [f"{row['ingredients']} {row['instructions']}" for _, row in self.df.iterrows()]
        self.embeddings = self.sbert.encode(texts, show_progress_bar=True)
        
        # Train autoencoder on SBERT embeddings
        logger.info("Training autoencoder...")
        self.train_autoencoder(self.embeddings)
        
        # Generate reduced embeddings
        self.autoencoder.eval()
        with torch.no_grad():
            _, self.reduced_embeddings = self.autoencoder(torch.FloatTensor(self.embeddings))
        self.reduced_embeddings = self.reduced_embeddings.numpy()
        
        logger.info("Data preparation completed")
    
    def visualize_embeddings(self, filename_prefix):
        # Generate t-SNE visualizations for both original and reduced embeddings
        logger.info("Generating t-SNE visualizations...")
        
        # Original SBERT embeddings
        tsne_original = TSNE(n_components=2, random_state=42)
        tsne_result_original = tsne_original.fit_transform(self.embeddings)
        
        fig_original = px.scatter(
            x=tsne_result_original[:, 0],
            y=tsne_result_original[:, 1],
            hover_data=[self.df['name']],
            title='t-SNE visualization of SBERT embeddings'
        )
        fig_original.write_html(f"{filename_prefix}_sbert.html")
        
        # Reduced embeddings from autoencoder
        tsne_reduced = TSNE(n_components=2, random_state=42)
        tsne_result_reduced = tsne_reduced.fit_transform(self.reduced_embeddings)
        
        fig_reduced = px.scatter(
            x=tsne_result_reduced[:, 0],
            y=tsne_result_reduced[:, 1],
            hover_data=[self.df['name']],
            title='t-SNE visualization of autoencoder-reduced embeddings'
        )
        fig_reduced.write_html(f"{filename_prefix}_reduced.html")
    
    def search(self, query_text: str, top_k: int = 10) -> List[dict]:
        # Generate embedding for query
        query_embedding = self.sbert.encode([query_text])
        
        # Reduce query embedding using autoencoder
        with torch.no_grad():
            _, reduced_query = self.autoencoder(torch.FloatTensor(query_embedding))
        reduced_query = reduced_query.numpy()
        
        # Calculate cosine similarities
        similarities = np.dot(self.reduced_embeddings, reduced_query.T).flatten()
        similarities = similarities / (np.linalg.norm(self.reduced_embeddings, axis=1) * np.linalg.norm(reduced_query))
        
        # Get top-k results
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

# FastAPI app
app = FastAPI(title="Recipe Search API with SBERT and Autoencoder")

class SearchResponse(BaseModel):
    results: List[dict]
    message: str

# Initialize search engine
search_engine = RecipeSearchEngine()

@app.on_event("startup")
async def startup_event():
    search_engine.prepare_data('recipes.csv')
    search_engine.visualize_embeddings('embeddings_viz')

@app.get("/search", response_model=SearchResponse)
async def search(query: str = Query(..., description="Search query text"), 
                limit: Optional[int] = Query(10, description="Number of results to return")):
    try:
        results = search_engine.search(query, top_k=limit)
        return SearchResponse(results=results, message="OK")
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=6352)
