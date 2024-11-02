# Recipe Search System with SBERT and Autoencoder
### Author: Arthur Cisotto Machado

## Setup Instructions

1. Create and activate Python virtual environment:
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the system:
```bash
python api.py
```

The system uses the included `recipes.csv` file. On first run, it:
- Generates embeddings from recipes
- Trains the autoencoder
- Creates visualizations

To retrain the model, delete the contents of `saved_models/` directory.
To generate new visualizations, delete the existing `.html` files.

## Data Collection (Optional)
The repository includes a pre-collected dataset. To collect fresh data:
```bash
python dataset_generator.py
```
This script scrapes 15,000 recipes from TudoGostoso website.

## Project Structure
- `recipes.csv`: Dataset with 15,000 recipes
- `dataset_generator.py`: Script for fresh data collection
- `models.py`: Implements denoising autoencoder
- `search_engine.py`: Processes embeddings and handles searches
- `api.py`: Main entry point and API
- `saved_models/`: Stores trained model and embeddings
- `embeddings_viz_sbert.html`: SBERT embeddings visualization
- `embeddings_viz_autoencoder.html`: Autoencoder-reduced embeddings visualization
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

# Step 1: Finding Embeddings

## Dataset Description
The dataset contains 15,000 recipes scraped from TudoGostoso website. Python scripts collect recipe data through CloudScraper. Each entry stores recipe name, ingredient list, and preparation instructions. The text exists in Portuguese format.

## Embedding Generation Process
SBERT creates vector representations from recipe texts. The process transforms recipe text into 768-dimensional vectors. A denoising autoencoder architecture processes these vectors. The network contains three encoding layers [512, 256, 128] with ReLU activations and BatchNorm.

```mermaid
graph LR
    subgraph Encoder
        I[Input 768] --> E1[Linear 512]
        E1 --> E1B[BatchNorm]
        E1B --> E1R[ReLU]
        E1R --> E1D[Dropout]
        E1D --> E2[Linear 256]
        E2 --> E2B[BatchNorm]
        E2B --> E2R[ReLU]
        E2R --> E2D[Dropout]
        E2D --> E3[Linear 128]
        E3 --> E3B[BatchNorm]
        E3B --> E3R[ReLU]
        E3R --> E3D[Dropout]
    end
    subgraph Decoder
        E3D --> D1[Linear 256]
        D1 --> D1B[BatchNorm]
        D1B --> D1R[ReLU]
        D1R --> D1D[Dropout]
        D1D --> D2[Linear 512]
        D2 --> D2B[BatchNorm]
        D2B --> D2R[ReLU]
        D2R --> D2D[Dropout]
        D2D --> D3[Linear 768]
        D3 --> D3B[BatchNorm]
        D3B --> D3R[ReLU]
        D3R --> D3D[Dropout]
    end
```

## Training Process
Mean Squared Error measures reconstruction quality between original and decoded vectors. The function suits the task through its sensitivity to component-wise differences in embeddings:

$$ L(x, \hat{x}) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x_i})^2 $$

The training applies random noise to input vectors. This process prevents overfitting through data augmentation. The optimization runs for 100 epochs with Adam optimizer.

# Step 2: Embedding Visualization

## SBERT Embedding Space
TSNE projects the original 768-dimensional SBERT embeddings into 2D space. The visualization reveals category-based groupings. Dessert recipes form a cluster on the right side. Meat-based recipes concentrate in the center, while rice and pasta recipes create smaller groups.

![SBERT Embeddings](embeddings_viz_sbert.html)

## Autoencoder-Reduced Space
The autoencoder reduces embeddings to 128 dimensions before TSNE projection. The reduced space maintains the global structure but shows less distinction between categories. The clusters appear more compressed. The Calinski-Harabasz score increases from 152.324 to 445.476, indicating stronger cluster separation. However, the negative Silhouette scores (-0.013 to -0.140) suggest overlap between categories.

## Visualization Analysis
The metrics reveal a trade-off in the dimensionality reduction process. The autoencoder creates more distinct cluster boundaries at the cost of category overlap. The SBERT embeddings maintain smoother transitions between recipe types. This difference impacts search behavior, where SBERT embeddings might provide more nuanced recipe relationships.

# Step 3: Search System Testing

## Test Cases Description
The system underwent three distinct search scenarios. The tests maintain consistency with APS1 queries. Each test evaluates different aspects of the embedding-based search system.

## Test Results

### Common Search: "frango" (Chicken)
The search returns 10 recipes with high relevance (scores 0.45-0.50). The results show recipe diversity.

```
Request URL
http://localhost:6352/search?query=frango
Server response
Code	Details
200	
Response body
Download
{
  "results": [
    {
      "title": "Frango a passarinho crocante na Air Fryer",
      "recipe_url": "https://www.tudogostoso.com.br/receita/169442-frango-a-passarinho-crocante-na-air-fryer.html",
      "ingredients": "400 g frango a passarinho temperado (5 pedaços), maizena para empanar",
      "similarity_score": 0.5058404207229614
    },
    {
      "title": "Filé de frango ao creme de milho",
      "recipe_url": "https://www.tudogostoso.com.br/receita/133999-file-de-frango-ao-creme-de-milho.html",
      "ingredients": "1 kg de filé de frango, 1 lata de milho com água, 1 requeijão, 1 creme de cebola, 1 creme de leite, 1 pacote de 50 g de queijo ralado, 250 g de mussarela",
      "similarity_score": 0.4720155894756317
    },
    {
      "title": "Peito de frango na panela de pressão",
      "recipe_url": "https://www.tudogostoso.com.br/receita/61024-peito-de-frango-na-panela-de-pressao.html",
      "ingredients": "2 cebolas médias, 2 dentes de alho, 3 colheres de óleo ou azeite, 1 peito de frango grande, Sal a gosto",
      "similarity_score": 0.46919170022010803
    },
    {
      "title": "Filé de frango crocante",
      "recipe_url": "https://www.tudogostoso.com.br/receita/96579-file-de-frango-crocante.html",
      "ingredients": "1 kg de filé de frango (sem osso), 3 ovos , Farinha de trigo , Pimenta, Sal, Limão, Sazón para carnes , Óleo (para fritar)",
      "similarity_score": 0.46390604972839355
    },
    {
      "title": "Caldo de pé de frango",
      "recipe_url": "https://www.tudogostoso.com.br/receita/300626-caldo-de-pe-de-frango.html",
      "ingredients": "1 kg de pé de frango, 1/2 kg de mandioca, alho, açafrão, cebola, 1 colher de óleo, 1 caldo de galinha, 1 pimenta bode, sal a gosto, cheiro verde",
      "similarity_score": 0.46225154399871826
    },
    {
      "title": "Pirão de frango",
      "recipe_url": "https://www.tudogostoso.com.br/receita/128360-pirao-de-frango.html",
      "ingredients": "2 peitos de frango , 1 cebola, 3 dentes de alho, 2 tomates, sal, caldo de frango, salsinha e cebolinha, farinha de mandioca",
      "similarity_score": 0.4606887996196747
    },
    {
      "title": "Filé de frango recheado",
      "recipe_url": "https://www.tudogostoso.com.br/receita/99483-file-de-frango-recheado.html",
      "ingredients": "4 filés de frango , 6 dentes de alho, 1 limão, tempero pronto para aves (opcional), 100 g de bacon, 1 pimentão vermelho grande, 1 cebola média, 1 tomate grande, 250 ml de caldo de galinha, 4 colheres de azeite, pimenta calabresa, sal, linha para culinária",
      "similarity_score": 0.456826388835907
    },
    {
      "title": "Frango com quiabo",
      "recipe_url": "https://www.tudogostoso.com.br/receita/20875-frango-com-quiabo.html",
      "ingredients": "1 kg de frango, limpo e cortado a passarinho, 2 colheres sopa de óleo, 1 cebola ralada, 2 dentes de alho amassado, Pimenta do reino, cheiro verde picadinho a gosto, 400g de quiabo picado, Sal a gosto",
      "similarity_score": 0.45619097352027893
    },
    {
      "title": "Nuggets caseiro",
      "recipe_url": "https://www.tudogostoso.com.br/receita/863-nuggets-caseiro.html",
      "ingredients": "1 kg de peito de frango (sem pele e sem osso), 2 ovos, 2 tablete de caldo de galinha, Farinha de rosca, Orégano e sal a gosto, 1 litro de água , Óleo",
      "similarity_score": 0.4534013867378235
    },
    {
      "title": "Caldo de frango",
      "recipe_url": "https://www.tudogostoso.com.br/receita/28535-caldo-de-frango.html",
      "ingredients": "1 peito de frango, 2 cubos de caldo knorr de frango, 1 cebola, 1 pimentão verde, 2 dentes de alho, 1 maço de cheiro verde, 500 g de mandioca, 2 pimentas de cheiro",
      "similarity_score": 0.4526980221271515
    }
  ],
  "message": "OK"
}
```

### Specific Search: "polvo" (Octopus)
The system returns 4 recipes with lower scores (0.34-0.37). The results demonstrate precision. The system avoids false positives and provides relevant octopus recipes.

```
Request URL
http://localhost:6352/search?query=polvo
Server response
Code	Details
200	
Response body
Download
{
  "results": [
    {
      "title": "Como cozinhar polvo",
      "recipe_url": "https://www.tudogostoso.com.br/receita/88233-como-cozinhar-polvo.html",
      "ingredients": "1 polvo, quanto maior melhor, mínimo 2,5 kg, 1 cebola inteira média, 2 folhas de louro, 1 copo de água, mais ou menos 220 ml",
      "similarity_score": 0.3787407875061035
    },
    {
      "title": "Arroz de polvo",
      "recipe_url": "https://www.tudogostoso.com.br/receita/1810-arroz-de-polvo.html",
      "ingredients": "2 polvos de 750 g, 1 cebola grande sem casca, 4 xícaras de chá de arroz parboirizado (lavado), 1 molhe de coentro, 1 molhe de cheiro verde, 1 dente de alho descascado e cortado, 2 colheres de sopa de azeite virgem, 3 Xícaras de água",
      "similarity_score": 0.36565449833869934
    },
    {
      "title": "Polvo na manteiga",
      "recipe_url": "https://www.tudogostoso.com.br/receita/179349-polvo-na-manteiga.html",
      "ingredients": "1 polvo, 3 cebolas, 1 pimentão, azeite, pimenta-do-reino, sal, manteiga",
      "similarity_score": 0.3571753203868866
    },
    {
      "title": "Polvo assado à moda do Porto",
      "recipe_url": "https://www.tudogostoso.com.br/receita/104819-polvo-assado-a-moda-do-porto.html",
      "ingredients": "1 polvo grande , 8 dentes de alho, 4 batatas grandes cortada ao meio, 2 copos de azeite de oliva, Sal a gosto",
      "similarity_score": 0.342401385307312
    }
  ],
  "message": "OK"
}
```

### Non-Obvious Search: "café" (Coffee)
The query returns 10 recipes (scores 0.48-0.55). The results show improvement from APS1:
- Direct coffee-based recipes (creamy coffee, espresso)
- Coffee-flavored desserts (brigadeiro)
- Coffee drink variations (cappuccino)
- No false positives from "coffee spoon" measurements, which appeared in APS1

```
Request URL
http://localhost:6352/search?query=caf%C3%A9
Server response
Code	Details
200	
Response body
Download
{
  "results": [
    {
      "title": "Café cremoso",
      "recipe_url": "https://www.tudogostoso.com.br/receita/164458-cafe-cremoso.html",
      "ingredients": "50 g de café solúvel, 200 ml de água fervendo, 200 g de açúcar, 1 litro de leite, chocolate em pó, canela em pó",
      "similarity_score": 0.5552593469619751
    },
    {
      "title": "Café expresso",
      "recipe_url": "https://www.tudogostoso.com.br/receita/63515-cafe-expresso.html",
      "ingredients": "50 g de café solúvel, 2 xícaras de açúcar refinado, 1 xícara de agua filtrada, Leite",
      "similarity_score": 0.5436257719993591
    },
    {
      "title": "Dalgona coffee, o café do TikTok",
      "recipe_url": "https://www.tudogostoso.com.br/receita/306516-dalgona-coffee-o-cafe-do-tiktok.html",
      "ingredients": "2 colheres (sopa) de água bem quente, 2 colheres (sopa) de açúcar, 2 colheres (sopa) de café solúvel, gelo, leite",
      "similarity_score": 0.525077223777771
    },
    {
      "title": "Brigadeiro de Café",
      "recipe_url": "https://www.tudogostoso.com.br/receita/93072-brigadeiro-de-cafe.html",
      "ingredients": "1 lata de leite condensado, 1 colher (sobremesa) rasa de café, 1 colher (sobremesa) cheia de chocolate em pó, 60 g de margarina",
      "similarity_score": 0.5017543435096741
    },
    {
      "title": "Café cremoso",
      "recipe_url": "https://www.tudogostoso.com.br/receita/11486-cafe-cremoso.html",
      "ingredients": "50 g de café solúvel, 2 xícaras de açúcar refinado, 1 xícara de água",
      "similarity_score": 0.5011235475540161
    },
    {
      "title": "Café Tradicional",
      "recipe_url": "https://www.tudogostoso.com.br/receita/130652-cafe-tradicional.html",
      "ingredients": "3 xícaras de água, 3 colheres (sopa) cheias de açúcar, 3 colheres (sopa) de pó de café toko",
      "similarity_score": 0.4892129600048065
    },
    {
      "title": "Café cremoso",
      "recipe_url": "https://www.tudogostoso.com.br/receita/21978-cafe-cremoso.html",
      "ingredients": "1 pacotinho de nescafé tradicional de 50 g, 2 xícaras de açúcar refinado, 1 xícara de água",
      "similarity_score": 0.4859789311885834
    },
    {
      "title": "Cappuccino fácil e econômico",
      "recipe_url": "https://www.tudogostoso.com.br/receita/73584-cappuccino-facil-e-economico.html",
      "ingredients": "1 medida de café solúvel, 2 medidas de açúcar (quem gostar de mais doce, coloque 3 medidas), 1 medida de água fervente",
      "similarity_score": 0.4839651584625244
    },
    {
      "title": "Café capuccino da Jaque",
      "recipe_url": "https://www.tudogostoso.com.br/receita/51790-cafe-capuccino-da-jaque.html",
      "ingredients": "200 g de leite em pó, 50 g de café solúvel, 50 g de açúcar cristal, 2 colheres de sopa de canela em pó, 2 colheres de café de bicarbonato de sódio",
      "similarity_score": 0.48277056217193604
    },
    {
      "title": "Creme de café (café cremoso)",
      "recipe_url": "https://www.tudogostoso.com.br/receita/123427-creme-de-cafe-cafe-cremoso.html",
      "ingredients": "50 g de café solúvel, 100 g de açúcar (pode utilizar o pacotinho do café como medidor), 200 ml de água quente",
      "similarity_score": 0.47817060351371765
    }
  ],
  "message": "OK"
}
```

## Search Behavior Analysis
The embedding-based system shows different characteristics from APS1. It captures semantic relationships without keyword-based limitations. The scores reflect both ingredient matches and recipe context similarity.

