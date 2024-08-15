from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
import uvicorn
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)  
    text = re.sub(r'[^\w\s]', '', text) 
    text = ' '.join(word for word in text.split() if word not in stop_words)
    units = ['kg', 'g', 'l', 'ml', 'xícara', 'colher', 'dente', 'pessoa', 'xícaras', 'colheres', 'copo', 'copos', 'unidade', 'unidades']
    text = ' '.join(word for word in text.split() if word not in units)
    return text

# Carregar o dataset
df = pd.read_csv('recipes.csv')

# Preprocessar os ingredientes
df['processed_ingredients'] = df['ingredients'].apply(preprocess_text)

# Inicializar o TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_ingredients'])

app = FastAPI()

class QueryResponse(BaseModel):
    title: str
    recipe_url: str
    content: str
    relevance: float

@app.get("/query", response_model=dict)
async def query(query_text: str = Query(..., description="Texto da consulta")):
    if not query_text:
        raise HTTPException(status_code=400, detail="Query parameter is missing")
    
    # Preprocessar o texto da consulta
    query_text_processed = preprocess_text(query_text)
    
    # Transformar a consulta usando o TF-IDF Vectorizer
    query_vec = vectorizer.transform([query_text_processed])
    
    # Calcular similaridade
    similarities = cosine_similarity(query_vec, X).flatten()
    
    # Obter os índices dos documentos mais relevantes
    indices = similarities.argsort()[-10:][::-1]
    indices = [i for i in indices if similarities[i] > 0]

    results = []
    for i in indices:
        results.append({
            'title': df.iloc[i]['name'],
            'recipe_url': df.iloc[i]['url'],
            'ingredients': df.iloc[i]['ingredients'][:500],
            'relevance': float(similarities[i])
        })
    
    return {"results": results, "message": "OK"}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8888)
