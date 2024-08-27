#!/bin/bash

# Passo 1: Criar um ambiente virtual
echo "Criando ambiente virtual..."
python3 -m venv venv

# Passo 2: Ativar o ambiente virtual
echo "Ativando ambiente virtual..."
source venv/bin/activate

# Passo 3: Baixar e instalar dependências
echo "Baixando e instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt

# Passo 4: Verificar se o dataset já existe
if [ ! -f "recipes.csv" ]; then
    echo "Arquivo recipes.csv não encontrado. Rodando o gerador de dataset..."
    python dataset_generator.py
else
    echo "Arquivo recipes.csv já existe. Pulando a geração do dataset."
fi

# Passo 5: Rodar o aplicativo FastAPI
echo "Rodando o aplicativo FastAPI..."
python app.py
