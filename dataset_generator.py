import csv
import cloudscraper
from bs4 import BeautifulSoup
import json
import resource

def get_page_content(url):
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'desktop': True,
            'mobile': False,
            'platform': 'windows'
        }
    )
    try:
        response = scraper.get(url)
        return response.text
    except Exception as e:
        print(f"Erro ao acessar a página: {e}")
        return None

def fetch_recipe(url):
    response = get_page_content(url)
    if response:
        soup = BeautifulSoup(response, 'html.parser')
        script_tags = soup.find_all('script', type='application/ld+json')
        for script in script_tags:
            try:
                data = json.loads(script.string)
                if data.get('@type') == 'Recipe':
                    return data
            except json.JSONDecodeError as e:
                print(f"Erro ao decodificar JSON: {e}")
                continue
    return None

def fetch_recipe_urls_from_page(page_url):
    urls = []
    response = get_page_content(page_url)
    if response:
        soup = BeautifulSoup(response, 'html.parser')
        recipe_links = soup.find_all('a', href=True)
        for link in recipe_links:
            href = link['href']
            if href.startswith('/receita/'):
                urls.append(f'https://www.tudogostoso.com.br{href}')
    return urls

def save_recipes_to_csv(recipes, filename="recipes.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'name', 'url', 'image_url', 'author', 'date_published', 'description',
            'rating_value', 'rating_count', 'keywords', 'prep_time', 'cook_time',
            'total_time', 'recipe_yield', 'recipe_category', 'ingredients', 'instructions'
        ])
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
    base_url = "https://www.tudogostoso.com.br/receitas?page="
    visited_urls = set()
    recipes = []
    
    while len(recipes) < max_recipes:
        page_url = base_url + str(start_page)
        print(f"Visiting page: {page_url}")
        recipe_urls = fetch_recipe_urls_from_page(page_url)
        
        for url in recipe_urls:
            if len(recipes) >= max_recipes:
                break
            if url not in visited_urls:
                visited_urls.add(url)
                print(f"Visiting: {url}")
                recipe_data = fetch_recipe(url)
                if recipe_data:
                    recipes.append(recipe_data)
                    print(f"Recipe saved: {recipe_data.get('name')}")
        
        start_page += 1 
    
    save_recipes_to_csv(recipes)
    print(f"Total recipes saved: {len(recipes)}")


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_NOFILE, (100000000, 100000000))
    dataset_generator(start_page=1, max_recipes=10000)
