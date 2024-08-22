# Sistema de Recomendação de Receitas

Desenvolvido por Arthur Cisotto Machado

## Sobre o Projeto

Este projeto implementa um sistema de recomendação de receitas utilizando uma API FastAPI. O sistema permite que os usuários busquem receitas com base em ingredientes, retornando as receitas mais relevantes.

### Caso de Uso

O sistema foi projetado para auxiliar na descoberta de receitas baseadas em ingredientes disponíveis ou preferências culinárias. É particularmente útil para:

1. Encontrar receitas com ingredientes específicos
2. Explorar novas ideias culinárias
3. Otimizar o uso de ingredientes disponíveis

## Conjunto de Dados

O conjunto de dados utilizado neste projeto foi coletado do site "TudoGostoso" (https://www.tudogostoso.com.br/), um dos maiores repositórios de receitas em português do Brasil. A escolha deste conjunto de dados se deve à sua relevância para o público-alvo e à variedade de receitas disponíveis.

Um script de web scraping foi desenvolvido para coletar informações sobre diversas receitas, incluindo nome, ingredientes, instruções e tempo de preparo. O script `dataset_generator.py` é responsável por esta coleta de dados.

É possível aumentar o tamanho do dataset modificando o parâmetro `max_recipes` na chamada da função `dataset_generator` no final do script:

```python
if __name__ == "__main__":
    dataset_generator(start_page=1, max_recipes=10000)
```

Aumentar o valor de `max_recipes` resultará em um conjunto de dados maior, o que pode melhorar a diversidade e a qualidade das recomendações. No entanto, isso também aumentará o tempo necessário para gerar o dataset e o espaço de armazenamento requerido.

## Instalação

Para instalar e configurar o projeto, siga estas etapas:

1. Clone o repositório:
   ```
   git clone [URL_DO_REPOSITORIO]
   cd APIRecomendacaoDeReceitas
   ```

2. Execute o script de instalação e configuração:
   ```
   chmod +x setup_and_run.sh
   ./setup_and_run.sh
   ```

   Este script irá:
   - Criar e ativar um ambiente virtual
   - Instalar as dependências necessárias
   - Gerar o dataset de receitas
   - Iniciar o servidor FastAPI

## Execução

Após a instalação inicial, inicie o servidor com o comando:

```
python app.py
```

O servidor estará disponível em `http://10.103.0.28:6352` (ou na porta especificada).

## Utilização

Para realizar uma busca, envie uma requisição GET para a rota `/query` com o parâmetro `query_text`. Exemplo:

```
http://10.103.0.28:6352/query?query_text=cenoura
```
## Formato de Retorno da API

Quando você faz uma consulta à API, ela retorna um objeto JSON contendo uma lista de receitas relevantes. Vamos usar como exemplo o resultado de uma busca pela query "frango":

```json
{
  "results": [
    {
      "title": "Panqueca de frango",
      "recipe_url": "https://www.tudogostoso.com.br/receita/6724-panqueca-de-frango.html",
      "ingredients": "3 ovos, 1 1/2 xícara (chá) de leite, 1 1/2 xícara (chá) de farinha de trigo, 1 colher de óleo, 2 cubinhos de caldo de frango, queijo ralado a gosto, 1 peito de frango cozido e desfiado, 3 colheres de massa de tomate, 3 colheres de cebola picadinho, sal a gosto",
      "relevance": 0.41278297837608346
    },
    // ... mais resultados ...
  ],
  "message": "OK"
}
```

Cada receita no array `results` contém as seguintes informações:

- `title`: O nome da receita.
- `recipe_url`: O link para a receita completa no site TudoGostoso.
- `ingredients`: Uma lista dos ingredientes necessários para a receita.
- `relevance`: Um valor numérico que indica quão relevante a receita é para a query fornecida. Quanto maior o valor, mais relevante é a receita.

A API retorna até 10 resultados, ordenados por relevância. O campo `message` indica o status da requisição, sendo "OK" para requisições bem-sucedidas.

A relevância é calculada usando técnicas de processamento de linguagem natural: os ingredientes e a query são convertidos em vetores TF-IDF, e a similaridade do cosseno entre esses vetores determina a relevância. Este método considera a frequência e importância das palavras, permitindo uma comparação eficaz entre a query e as receitas no banco de dados.

## Exemplos de Testes

1. Teste que retorna 10 resultados:
   [http://10.103.0.28:6352/query?query_text=frango](http://10.103.0.28:6352/query?query_text=frango)
   
   Este teste retorna 10 resultados devido à popularidade e versatilidade do frango em diversas receitas.

2. Teste que retorna menos de 10 resultados:
   [http://10.103.0.28:6352/query?query_text=polvo](http://10.103.0.28:6352/query?query_text=polvo)
   
   Este teste provavelmente retornará menos de 10 resultados devido à especificidade do ingrediente.

3. Teste que retorna algo não óbvio:
   [http://10.103.0.28:6352/query?query_text=café](http://10.103.0.28:6352/query?query_text=café)
     
   Este teste busca por receitas que utilizam café como ingrediente, mas revela um aspecto interessante do sistema de busca. Muitas receitas aparecem devido ao uso de "café" como unidade de medida (por exemplo, "1 colher de café de canela"). Isso demonstra uma limitação atual do pré-processamento de texto, que já remove unidades de medida comuns como "colher" e "xícara", mas não considera "café" nesse contexto. Esta descoberta oferece uma oportunidade de melhoria no algoritmo de busca, destacando como resultados aparentemente não relacionados podem surgir devido a nuances linguísticas em receitas.

