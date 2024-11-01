import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    def categorize_recipe(self, name: str, ingredients: str, instructions: str) -> List[str]:
        """
        Categorizes a recipe based on its name, ingredients and instructions.
        Returns a list of applicable categories.
        """
        name = name.lower()
        ingredients = ingredients.lower()
        instructions = instructions.lower()
        
        # Dicionário de categorias e suas palavras-chave associadas
        category_keywords = {
            'Chocolate': [
                'chocolate', 'cacau', 'brigadeiro', 'brownie', 'mousse de chocolate',
                'chocolate em pó', 'achocolatado'
            ],
            'Meat': [
                'carne', 'bife', 'picanha', 'alcatra', 'costela', 'cupim', 'maminha',
                'patinho', 'contrafilé', 'filé mignon'
            ],
            'Chicken': [
                'frango', 'galinha', 'coxa', 'sobrecoxa', 'peito de frango',
                'asa de frango', 'coxinha da asa'
            ],
            'Fish': [
                'peixe', 'salmão', 'atum', 'bacalhau', 'tilápia', 'sardinha',
                'pescada', 'merluza', 'camarão', 'polvo', 'lula'
            ],
            'Pork': [
                'porco', 'bacon', 'linguiça', 'costela de porco', 'pernil',
                'lombo', 'pancetta', 'calabresa'
            ],
            'Pasta': [
                'macarrão', 'espaguete', 'penne', 'lasanha', 'nhoque',
                'ravioli', 'talharim', 'fetuccine'
            ],
            'Rice': [
                'arroz', 'risoto', 'arroz integral', 'arroz carreteiro',
                'arroz de forno'
            ],
            'Beans': [
                'feijão', 'feijoada', 'lentilha', 'grão de bico', 'ervilha'
            ],
            'Salad': [
                'salada', 'alface', 'rúcula', 'agrião', 'couve'
            ],
            'Cake': [
                'bolo', 'torta', 'cupcake', 'muffin', 'rocambole'
            ],
            'Cookie': [
                'biscoito', 'cookie', 'bolacha'
            ],
            'Pie': [
                'torta', 'quiche', 'empadão'
            ],
            'Bread': [
                'pão', 'broa', 'brioche', 'focaccia', 'ciabatta'
            ]
        }
        
        # Lista para armazenar as categorias encontradas
        found_categories = set()
        
        # Verifica cada categoria
        for category, keywords in category_keywords.items():
            # Procura keywords no nome, ingredientes e instruções
            for keyword in keywords:
                # Evita matches parciais usando espaços
                if f' {keyword} ' in f' {name} ' or \
                f' {keyword} ' in f' {ingredients} ' or \
                f' {keyword} ' in f' {instructions} ':
                    found_categories.add(category)
                    break
        
        # Adiciona categorias especiais baseadas em combinações
        all_text = f'{name} {ingredients} {instructions}'
        
        # Detecta pratos vegetarianos (não contém carne)
        meat_keywords = set()
        for category in ['Meat', 'Chicken', 'Fish', 'Pork']:
            meat_keywords.update(category_keywords[category])
        
        has_meat = any(f' {keyword} ' in f' {all_text} ' for keyword in meat_keywords)
        if not has_meat and any(keyword in all_text for keyword in ['legume', 'verdura', 'vegano', 'vegetariano']):
            found_categories.add('Vegetarian')
        
        # Detecta sobremesas
        dessert_categories = {'Chocolate', 'Cake', 'Cookie', 'Pie'}
        sweet_ingredients = {'açúcar', 'chocolate', 'leite condensado', 'doce de leite', 'mel'}
        if (found_categories & dessert_categories) or \
        any(ingredient in ingredients for ingredient in sweet_ingredients):
            found_categories.add('Dessert')
        
        # Se nenhuma categoria foi encontrada, marca como Other
        if not found_categories:
            found_categories.add('Other')
        
        return list(found_categories)

    def visualize_embeddings(self, filename_prefix):
        """
        Visualizes and compares both original SBERT embeddings and reduced autoencoder embeddings
        using TSNE and includes clustering analysis.
        """
        sbert_viz_path = f"{filename_prefix}_sbert.html"
        reduced_viz_path = f"{filename_prefix}_reduced.html"
        comparison_viz_path = f"{filename_prefix}_comparison.html"
        
        logger.info("Generating TSNE visualizations...")
        
        # TSNE for original SBERT embeddings
        tsne_original = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_result_original = tsne_original.fit_transform(self.embeddings)
        
        # TSNE for autoencoder-reduced embeddings
        tsne_reduced = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_result_reduced = tsne_reduced.fit_transform(self.reduced_embeddings)
        
        # Get categories for each recipe
        categories_list = [
            self.categorize_recipe(
                name=row['name'],
                ingredients=row['ingredients'],
                instructions=row['instructions']
            )
            for _, row in self.df.iterrows()
        ]
        
        # Create main category for visualization (using first category)
        main_categories = [cats[0] if cats else 'Other' for cats in categories_list]
        
        # Create DataFrame for plotting
        plot_df_original = pd.DataFrame({
            'x': tsne_result_original[:, 0],
            'y': tsne_result_original[:, 1],
            'name': self.df['name'],
            'main_category': main_categories,
            'all_categories': [', '.join(cats) for cats in categories_list],
            'ingredients': self.df['ingredients']
        })
        
        plot_df_reduced = pd.DataFrame({
            'x': tsne_result_reduced[:, 0],
            'y': tsne_result_reduced[:, 1],
            'name': self.df['name'],
            'main_category': main_categories,
            'all_categories': [', '.join(cats) for cats in categories_list],
            'ingredients': self.df['ingredients']
        })
        
        # Create interactive plots
        fig_original = px.scatter(
            plot_df_original,
            x='x',
            y='y',
            color='main_category',
            hover_data=['name', 'all_categories', 'ingredients'],
            title='SBERT Embeddings Visualization',
            labels={'main_category': 'Main Category'}
        )
        
        fig_reduced = px.scatter(
            plot_df_reduced,
            x='x',
            y='y',
            color='category',
            hover_data=['name', 'ingredients'],
            title='Autoencoder-Reduced Embeddings Visualization',
            labels={'color': 'Recipe Category'}
        )
        
        # Create comparison plot
        fig_comparison = make_subplots(
            rows=1, cols=2,
            subplot_titles=('SBERT Embeddings', 'Autoencoder-Reduced Embeddings')
        )
        
        # Add traces for each category
        for category in plot_df_original['category'].unique():
            # Original embeddings
            mask_orig = plot_df_original['category'] == category
            fig_comparison.add_trace(
                go.Scatter(
                    x=plot_df_original[mask_orig]['x'],
                    y=plot_df_original[mask_orig]['y'],
                    name=category,
                    mode='markers',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Reduced embeddings
            mask_red = plot_df_reduced['category'] == category
            fig_comparison.add_trace(
                go.Scatter(
                    x=plot_df_reduced[mask_red]['x'],
                    y=plot_df_reduced[mask_red]['y'],
                    name=category,
                    mode='markers',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig_comparison.update_layout(
            title_text="Comparison of Embedding Spaces",
            height=600
        )
        
        # Save visualizations
        fig_original.write_html(sbert_viz_path)
        fig_reduced.write_html(reduced_viz_path)
        fig_comparison.write_html(comparison_viz_path)
        
        logger.info("Visualizations generated successfully")
        
        # Calculate and log clustering metrics
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        # Convert categories to numerical labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels = le.fit_transform(categories)
        
        # Calculate clustering metrics for both embedding spaces
        sbert_silhouette = silhouette_score(self.embeddings, labels)
        reduced_silhouette = silhouette_score(self.reduced_embeddings, labels)
        
        sbert_ch = calinski_harabasz_score(self.embeddings, labels)
        reduced_ch = calinski_harabasz_score(self.reduced_embeddings, labels)
        
        logger.info(f"""
        Clustering Metrics:
        SBERT Embeddings:
        - Silhouette Score: {sbert_silhouette:.3f}
        - Calinski-Harabasz Score: {sbert_ch:.3f}
        
        Autoencoder-Reduced Embeddings:
        - Silhouette Score: {reduced_silhouette:.3f}
        - Calinski-Harabasz Score: {reduced_ch:.3f}
        """)

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