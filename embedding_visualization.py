import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
import torch
import re
from tqdm import tqdm

def extract_main_ingredients(ingredients_text):
    """Extract key ingredients for categorization."""
    common_ingredients = {
        'carne': 'Carnes',
        'frango': 'Aves',
        'peixe': 'Peixes',
        'camarão': 'Frutos do Mar',
        'chocolate': 'Doces',
        'açúcar': 'Doces',
        'arroz': 'Básicos',
        'feijão': 'Básicos',
        'salada': 'Vegetariano',
        'vegetais': 'Vegetariano',
        'massa': 'Massas'
    }
    
    ingredients_text = ingredients_text.lower()
    for key, category in common_ingredients.items():
        if key in ingredients_text:
            return category
    return 'Outros'

def create_embeddings_visualization(df, custom_embeddings, sbert_model=None):
    """Create and compare t-SNE visualizations for both embedding types."""
    # Generate SBERT embeddings if not provided
    if sbert_model is None:
        sbert_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    
    print("Generating SBERT embeddings for comparison...")
    base_embeddings = sbert_model.encode(df['ingredients'].tolist(), show_progress_bar=True)
    
    # Add recipe categories based on ingredients
    df['category'] = df['ingredients'].apply(extract_main_ingredients)
    
    # Generate t-SNE projections
    print("Generating t-SNE projections...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    custom_tsne = tsne.fit_transform(custom_embeddings)
    base_tsne = tsne.fit_transform(base_embeddings)
    
    # Create interactive visualizations
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Custom Embeddings', 'SBERT Embeddings'),
                       specs=[[{'type': 'scatter'}, {'type': 'scatter'}]])
    
    # Custom embeddings plot
    for category in df['category'].unique():
        mask = df['category'] == category
        fig.add_trace(
            go.Scatter(
                x=custom_tsne[mask, 0],
                y=custom_tsne[mask, 1],
                mode='markers',
                name=category,
                text=df[mask]['name'],
                hovertemplate="<b>%{text}</b><br>" +
                             "Category: " + category +
                             "<extra></extra>",
                showlegend=True
            ),
            row=1, col=1
        )
    
    # SBERT embeddings plot
    for category in df['category'].unique():
        mask = df['category'] == category
        fig.add_trace(
            go.Scatter(
                x=base_tsne[mask, 0],
                y=base_tsne[mask, 1],
                mode='markers',
                name=category,
                text=df[mask]['name'],
                hovertemplate="<b>%{text}</b><br>" +
                             "Category: " + category +
                             "<extra></extra>",
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1600,
        title_text="Comparison of Recipe Embeddings Clustering",
        hovermode='closest'
    )
    
    # Save visualization
    fig.write_html("embedding_visualization.html")
    
    # Analyze clustering metrics
    print("\nClustering Analysis:")
    print("Custom Embeddings Cluster Distribution:")
    custom_cluster_dist = df['category'].value_counts()
    print(custom_cluster_dist)
    
    return fig, custom_cluster_dist

def analyze_nearby_recipes(df, embeddings, recipe_index, n_neighbors=5):
    """Analyze recipes near a given recipe in the embedding space."""
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(embeddings)
    distances, indices = nbrs.kneighbors([embeddings[recipe_index]])
    
    print(f"\nSimilar recipes to '{df.iloc[recipe_index]['name']}':")
    for idx, distance in zip(indices[0][1:], distances[0][1:]):
        print(f"- {df.iloc[idx]['name']} (Distance: {distance:.3f})")

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('recipes.csv')
    custom_embeddings = np.load('recipe_embeddings.npy')
    
    # Create visualization
    fig, cluster_dist = create_embeddings_visualization(df, custom_embeddings)
    
    # Analyze some example recipes
    print("\nAnalyzing example recipes...")
    for _ in range(3):
        random_idx = np.random.randint(len(df))
        analyze_nearby_recipes(df, custom_embeddings, random_idx)
