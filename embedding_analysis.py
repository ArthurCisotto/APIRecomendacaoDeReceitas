import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, silhouette_score, calinski_harabasz_score
import umap
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import pickle
import os
import json
import warnings
import shutil

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class EmbeddingAnalyzer:
    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.embeddings = search_engine.embeddings
        self.latent_embeddings = search_engine.latent_embeddings
        
        self.cache_dir = "analysis_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.results_path = os.path.join(self.cache_dir, "embedding_results.json")
        self.plot_path = "embedding_comparison.png"
        
        self.mlb = MultiLabelBinarizer()
        self._init_categories()

    def _init_categories(self):
        """Initialize categories and their binary representation"""
        logger.info("Generating categories...")
        self.categories = []
        for _, row in self.search_engine.df.iterrows():
            cats = self.search_engine.categorize_recipe(
                row['name'],
                row['ingredients'],
                row['instructions']
            )
            self.categories.append(cats)
        
        # Convert to binary format
        self.binary_categories = self.mlb.fit_transform(self.categories)
        logger.info("Categories generated successfully")

    def reduce_pca(self):
        """Reduce dimensionality using PCA"""
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(self.embeddings)
        pca = PCA(n_components=128)
        return pca.fit_transform(normalized_embeddings)
    
    def reduce_umap(self):
        """Reduce dimensionality using UMAP"""
        # Normalizar os dados primeiro
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(self.embeddings)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            reducer = umap.UMAP(
                n_components=128,
                random_state=42,
                n_jobs=1,  
                n_neighbors=15,  
                min_dist=0.1,
                metric='euclidean'
            )
            return reducer.fit_transform(normalized_embeddings)
        
    def evaluate_embeddings(self, embeddings, name=""):
        """Calculate metrics for given embeddings"""
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(embeddings)
        
        X_train, X_test, y_train, y_test = train_test_split(
            normalized_embeddings,
            self.binary_categories,
            test_size=0.2,
            random_state=42
        )
        
        clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                tol=1e-4,
                solver='lbfgs',
                n_jobs=-1
            )
        )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        silhouette = silhouette_score(normalized_embeddings, np.argmax(self.binary_categories, axis=1))
        calinski = calinski_harabasz_score(normalized_embeddings, np.argmax(self.binary_categories, axis=1))
        
        logger.info(f"\nResults for {name}:")
        logger.info(f"Classification F1: {f1:.3f}")
        logger.info(f"Silhouette Score: {silhouette:.3f}")
        logger.info(f"Calinski-Harabasz Score: {calinski:.3f}")
        
        return {
            'name': name,
            'f1': float(f1),
            'silhouette': float(silhouette),
            'calinski': float(calinski)
        }
    
    def compare_methods(self, force_recompute=False):
        """Compare different embedding methods"""
        results = []
        
        logger.info("\nEvaluating SBERT Base embeddings...")
        results.append(self.evaluate_embeddings(self.embeddings, "SBERT Base"))
        
        logger.info("\nEvaluating Autoencoder embeddings...")
        results.append(self.evaluate_embeddings(self.latent_embeddings, "Autoencoder"))
        
        logger.info("\nApplying and evaluating PCA...")
        pca_embeddings = self.reduce_pca()
        results.append(self.evaluate_embeddings(pca_embeddings, "PCA"))
        
        logger.info("\nApplying and evaluating UMAP...")
        umap_embeddings = self.reduce_umap()
        results.append(self.evaluate_embeddings(umap_embeddings, "UMAP"))
        
        with open(self.results_path, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder)
        
        return results
    
    def plot_comparison(self, results):
        """Create bar plot comparing methods with better visualization"""
        df = pd.DataFrame(results)
        melted_df = df.melt(
            id_vars=['name'],
            value_vars=['f1', 'silhouette', 'calinski'],
            var_name='Metric',
            value_name='Score'
        )
        
        melted_df.loc[melted_df['Metric'] == 'calinski', 'Score'] /= 1000
        
        # Configurar figura com fundo branco
        plt.figure(figsize=(14, 8), facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')
        
        # Cores mais distintas
        colors = {
            'f1': '#2ecc71',       # Verde
            'silhouette': '#3498db',  # Azul
            'calinski': '#e74c3c'    # Vermelho
        }
        
        # Configurar o gráfico
        bar_width = 0.25
        positions = np.arange(len(df))
        
        # Plotar cada métrica separadamente
        metric_names = {
            'f1': 'Classification F1',
            'silhouette': 'Silhouette Score',
            'calinski': 'Calinski Score (÷1000)'
        }
        
        for i, metric in enumerate(['f1', 'silhouette', 'calinski']):
            data = melted_df[melted_df['Metric'] == metric]
            bars = ax.bar(
                positions + i * bar_width,
                data['Score'],
                bar_width,
                label=metric_names[metric],
                color=colors[metric],
                alpha=0.8
            )
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar_width/2,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )
        
        # Configurar os eixos
        ax.set_xticks(positions + bar_width)
        ax.set_xticklabels(df['name'].unique(), rotation=45, ha='right')
        
        # Adicionar títulos e labels
        plt.title('Comparison of Embedding Methods', fontsize=14, pad=20, fontweight='bold')
        plt.xlabel('Method', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        
        # Melhorar a legenda
        plt.legend(
            title='Metrics',
            title_fontsize=12,
            fontsize=10,
            bbox_to_anchor=(1.15, 1),
            loc='upper left',
            borderaxespad=0.,
            frameon=True,
            edgecolor='black'
        )
        
        # Adicionar grid mais sutil
        plt.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
        
        # Remover bordas extras
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar com alta qualidade
        plt.savefig(
            self.plot_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close()
        
        logger.info(f"Plot saved to {self.plot_path}")

def run_analysis(search_engine):
    """Run the complete analysis"""
    analyzer = EmbeddingAnalyzer(search_engine)
    if os.path.exists(analyzer.plot_path):
        logger.info("Plot already exists, skipping...")
        return
    results = analyzer.compare_methods()
    analyzer.plot_comparison(results)
    return results