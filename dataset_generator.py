import csv
import cloudscraper
from bs4 import BeautifulSoup
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import time
from tqdm import tqdm
import os

def create_scraper():
    return cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'desktop': True,
            'mobile': False,
            'platform': 'windows'
        }
    )

def fetch_recipe_urls_from_page(page_number):
    base_url = f"https://www.tudogostoso.com.br/receitas?page={page_number}"
    urls = []
    scraper = create_scraper()
    
    try:
        response = scraper.get(base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            recipe_links = soup.find_all('a', href=True)
            for link in recipe_links:
                href = link['href']
                if href.startswith('/receita/'):
                    urls.append(f'https://www.tudogostoso.com.br{href}')
    except Exception as e:
        print(f"Erro ao coletar URLs da página {page_number}: {e}")
    
    return urls

def fetch_recipe_batch(urls):
    recipes = []
    scraper = create_scraper()
    
    for url in urls:
        try:
            response = scraper.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                script_tags = soup.find_all('script', type='application/ld+json')
                for script in script_tags:
                    try:
                        data = json.loads(script.string)
                        if data.get('@type') == 'Recipe':
                            data['url'] = url
                            recipes.append(data)
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Erro ao processar {url}: {e}")
    
    return recipes

def save_batch_to_csv(recipes, filename, lock):
    with lock:
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for recipe in recipes:
                writer.writerow([
                    recipe.get('name'),
                    recipe.get('url'),
                    recipe.get('image', {}).get('url', None),
                    recipe.get('author', {}).get('name', None),
                    recipe.get('datePublished'),
                    recipe.get('description'),
                    recipe.get('aggregateRating', {}).get('ratingValue', None),
                    recipe.get('aggregateRating', {}).get('ratingCount', None),
                    recipe.get('keywords'),
                    recipe.get('prepTime'),
                    recipe.get('cookTime'),
                    recipe.get('totalTime'),
                    recipe.get('recipeYield'),
                    recipe.get('recipeCategory'),
                    ', '.join(recipe.get('recipeIngredient', [])),
                    ' '.join(step.get('text', '') for step in recipe.get('recipeInstructions', []))
                ])

def dataset_generator(start_page=1, max_recipes=100):
    # Inicializar arquivo CSV
    with open("recipes.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'name', 'url', 'image_url', 'author', 'date_published', 'description',
            'rating_value', 'rating_count', 'keywords', 'prep_time', 'cook_time',
            'total_time', 'recipe_yield', 'recipe_category', 'ingredients', 'instructions'
        ])

    # Coletar URLs em paralelo
    print("Coletando URLs de receitas...")
    urls = set()
    pages_to_fetch = (max_recipes // 15) + 1  # Assumindo média de 15 receitas por página
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_recipe_urls_from_page, page) 
                  for page in range(start_page, start_page + pages_to_fetch)]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Coletando páginas"):
            urls.update(future.result())
    
    recipe_urls = list(urls)[:max_recipes]
    print(f"URLs coletadas: {len(recipe_urls)}")

    # Processar receitas em lotes
    batch_size = 50
    url_batches = [recipe_urls[i:i + batch_size] for i in range(0, len(recipe_urls), batch_size)]
    
    # Lock para escrita no arquivo
    file_lock = multiprocessing.Manager().Lock()
    
    recipes_processed = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as process_executor:
        futures = []
        for batch in url_batches:
            futures.append(process_executor.submit(fetch_recipe_batch, batch))
        
        with tqdm(total=len(recipe_urls), desc="Processando receitas") as pbar:
            for future in as_completed(futures):
                recipes_batch = future.result()
                if recipes_batch:
                    save_batch_to_csv(recipes_batch, "recipes.csv", file_lock)
                    recipes_processed += len(recipes_batch)
                    pbar.update(len(recipes_batch))
                
                if recipes_processed >= max_recipes:
                    break

    print(f"Total de receitas salvas: {recipes_processed}")

if __name__ == "__main__":
    start_time = time.time()
    
    multiprocessing.set_start_method('spawn', force=True)
    dataset_generator(start_page=1, max_recipes=15000)
    
    end_time = time.time()
    print(f"Tempo total de execução: {end_time - start_time:.2f} segundos")