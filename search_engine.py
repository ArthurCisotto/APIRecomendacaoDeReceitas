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
import logging
import os
import pickle
import re
from typing import List, Dict
from tqdm import tqdm
from models import DenoisingAutoencoder, RecipeDataset

logger = logging.getLogger(__name__)

class RecipeSearchEngine:
    def __init__(self, model_name='distiluse-base-multilingual-cased-v1'):
        self.sbert = SentenceTransformer(model_name)
        self.autoencoder = None
        self.df = None
        self.embeddings = None  # original embeddings do SBERT
        self.latent_embeddings = None  # embeddings do espaço latente (128d)
        
        # Caminhos para os arquivos salvos
        self.model_dir = "saved_models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.autoencoder_path = os.path.join(self.model_dir, "autoencoder.pt")
        self.embeddings_path = os.path.join(self.model_dir, "embeddings.pkl")
        self.latent_embeddings_path = os.path.join(self.model_dir, "latent_embeddings.pkl")
        
    def train_autoencoder(self, embeddings, epochs=100, batch_size=32):
        dataset = RecipeDataset(embeddings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.autoencoder = DenoisingAutoencoder(
            input_dim=512,  # dimensão do modelo multilíngue
            hidden_dims=[384, 256, 128]  # redução mais gradual
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
        with open(self.latent_embeddings_path, 'wb') as f:
            pickle.dump(self.latent_embeddings, f)
    
    def load_data(self):
        if os.path.exists(self.embeddings_path) and \
           os.path.exists(self.latent_embeddings_path) and \
           os.path.exists(self.autoencoder_path):
            logger.info("Carregando dados e modelo salvos...")
            
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            with open(self.latent_embeddings_path, 'rb') as f:
                self.latent_embeddings = pickle.load(f)
                
            self.autoencoder = DenoisingAutoencoder(input_dim=512)  # dim correta
            self.autoencoder.load_state_dict(torch.load(self.autoencoder_path))
            self.autoencoder.eval()
            
            return True
        return False
    
    def prepare_data(self, csv_path):
        """
        Prepara os dados carregando o CSV e gerando embeddings.
        """
        self.df = pd.read_csv(csv_path)
        
        if self.load_data():
            return
        
        logger.info("Gerando embeddings SBERT...")
        texts = []
        for _, row in self.df.iterrows():
            text = f"receita de {row['name']}. ingredientes: {row['ingredients']}. modo de preparo: {row['instructions']}"
            texts.append(text)
        
        # Gera embeddings - SBERT já retorna um embedding por sentença
        embeddings = []
        for text in tqdm(texts, desc="Gerando embeddings"):
            emb = self.sbert.encode(text)
            embeddings.append(emb)
        
        self.embeddings = np.array(embeddings)
        
        # Treina autoencoder
        logger.info("Treinando denoising autoencoder...")
        self.train_autoencoder(self.embeddings)
        
        # Gera embeddings do espaço latente
        self.autoencoder.eval()
        with torch.no_grad():
            _, self.latent_embeddings = self.autoencoder(
                torch.FloatTensor(self.embeddings),
                add_noise=False
            )
        self.latent_embeddings = self.latent_embeddings.numpy()
        
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
        sweet_ingredients = {'chocolate', 'leite condensado', 'doce de leite', 'mel'}
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
        
        if os.path.exists(sbert_viz_path) and os.path.exists(reduced_viz_path):
            logger.info("Visualizations already generated")
            return
        
        logger.info("Generating TSNE visualizations...")
        
        # TSNE for original SBERT embeddings
        tsne_original = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_result_original = tsne_original.fit_transform(self.embeddings)
        
        # TSNE for autoencoder-reduced embeddings
        tsne_reduced = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_result_reduced = tsne_reduced.fit_transform(self.latent_embeddings)
        
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
        
        # Create interactive plots with Plotly
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
            color='main_category',
            hover_data=['name', 'all_categories', 'ingredients'],
            title='Autoencoder-Reduced Embeddings Visualization',
            labels={'main_category': 'Main Category'}
        )

        # Save visualizations
        fig_original.write_html(sbert_viz_path)
        fig_reduced.write_html(reduced_viz_path)
        
        logger.info("Visualizations generated successfully")
        
        # Calculate and log clustering metrics
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        from sklearn.preprocessing import LabelEncoder
        
        # Convert categories to numerical labels
        le = LabelEncoder()
        labels = le.fit_transform(main_categories) 
        
        # Calculate clustering metrics for both embedding spaces
        sbert_silhouette = silhouette_score(self.embeddings, labels)
        reduced_silhouette = silhouette_score(self.latent_embeddings, labels)
        
        sbert_ch = calinski_harabasz_score(self.embeddings, labels)
        reduced_ch = calinski_harabasz_score(self.latent_embeddings, labels)
        
        logger.info(f"""
        Clustering Metrics:
        SBERT Embeddings:
        - Silhouette Score: {sbert_silhouette:.3f}
        - Calinski-Harabasz Score: {sbert_ch:.3f}
        
        Autoencoder-Reduced Embeddings:
        - Silhouette Score: {reduced_silhouette:.3f}
        - Calinski-Harabasz Score: {reduced_ch:.3f}
        """)

    def search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """
        Search for recipes using SBERT embeddings directly.
        """
        # SBERT já retorna um embedding por sentença
        query_embedding = self.sbert.encode(f"receita de {query_text}")
        
        # Calcular similaridades diretamente com os embeddings do SBERT
        similarities = np.dot(self.embeddings, query_embedding)
        similarities = similarities / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices[:top_k]:
            results.append({
                'title': self.df.iloc[idx]['name'],
                'recipe_url': self.df.iloc[idx]['url'],
                'ingredients': self.df.iloc[idx]['ingredients'],
                'similarity_score': float(similarities[idx])
            })
        
        # Filter out low similarity scores dynamically 
        results = [res for res in results if res['similarity_score'] > 0.5]
        
        return results