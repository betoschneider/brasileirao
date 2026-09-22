# Brasileirão Série A - Previsão de Resultados

## Sobre o Projeto
Este projeto utiliza técnicas de Ciência de Dados e Machine Learning para analisar e prever os resultados das partidas do Campeonato Brasileiro Série A. O sistema coleta dados de APIs públicas, processa estatísticas e aplica modelos preditivos para estimar a classificação final e os resultados rodada a rodada.

## Funcionalidades
- **Análise de Dados**: Processamento de dados históricos e atuais do campeonato.
- **Modelos de Machine Learning**: Utilização de algoritmos como Random Forest, Logistic Regression, Gradient Boosting, KNN e Voting Classifier.
- **Validação Robusta**: Técnicas de validação como Leave-One-Season-Out (LOSO) e Monte Carlo Cross-Validation (MCCV), que também estimam a confiança de cada previsão.
- **Interface Web (Streamlit)**:
    - Classificação final prevista e gráfico de pontos.
    - Probabilidade de título, G4 e rebaixamento (simulações Monte Carlo).
    - Evolução da posição de cada time rodada a rodada.
    - Previsões detalhadas das partidas, com a confiança de cada resultado.
    - Estatísticas de desempenho por time.

## Tecnologias Utilizadas
- **Linguagem**: Python 3.12+
- **Análise de Dados**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualização**: Altair, Streamlit
- **Gerenciamento de dependências**: [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`)
- **Containerização**: Docker

## Como Executar

### Pré-requisitos
- [uv](https://docs.astral.sh/uv/getting-started/installation/) instalado
- Docker (opcional)

### Instalação Local
1. Clone o repositório:
   ```bash
   git clone https://github.com/betoschneider/brasileirao.git
   cd brasileirao
   ```
2. Crie o ambiente e instale as dependências:
   ```bash
   uv sync
   ```
   Para adicionar uma nova dependência use `uv add <pacote>` (o `uv.lock` é atualizado automaticamente).
   O `uv sync` instala também o grupo `pipeline` (`scikit-learn`, `joblib`, `requests`), usado
   só por `atualizar_previsoes.py`; a imagem de produção é montada sem esse grupo.

### Atualizando as Previsões
O pipeline que antes era um notebook (`brasileirao-analise.ipynb`) agora é o script `atualizar_previsoes.py`:
```bash
uv run atualizar_previsoes.py
```
Opções úteis:
- `--n-splits 1000`: aumenta o número de simulações da validação Monte Carlo (padrão `200`),
  deixando a confiança de cada previsão mais fina porém a execução bem mais lenta
  (aproximadamente 1 hora).
- `--n-jobs 4`: limita os núcleos usados no processamento paralelo.

O script grava `app/tabela-evolucao-modelos.csv`, `app/partidas-modelos.csv`,
`app/times-stats.csv` e `app/chances-times.csv`, que são consumidos pelo dashboard.

> **Onde rodar**: a geração das previsões é pesada (a validação Monte Carlo usa todos os
> núcleos da máquina e vários processos). Por isso ela **não** roda no servidor: rode
> `atualizar_previsoes.py` na sua máquina e publique os CSVs gerados.

```bash
uv run atualizar_previsoes.py                       # atualiza os CSVs em app/
git add app/*.csv                                   # publica os resultados
git commit -m "dados: atualiza rodada 30"
git push
```

Como os CSVs ficam em `app/` e essa pasta é montada como volume no container, o push já é
suficiente: o Streamlit lê os arquivos a cada recarregamento da página, sem reiniciar nada.
O workflow de deploy detecta que só os dados mudaram e **não reconstrói a imagem**.

### Executando a Aplicação Web
```bash
uv run streamlit run app/app.py
```

### Usando Docker
1. Construa e inicie os containers:
   ```bash
   docker compose up --build
   ```
2. Acesse a aplicação em `http://localhost:8504`.

O container de produção apenas serve o dashboard. Como o projeto é montado em `/app`
(`.:/app`) e a imagem guarda o ambiente em `/opt/venv`, atualizar dados não exige rebuild.

## Estrutura do Projeto
- `atualizar_previsoes.py`: pipeline de ETL e modelagem (coleta, features, treino e projeção).
- `app/app.py`: dashboard Streamlit.
- `app/*.csv`: dados gerados pelo pipeline e consumidos pelo dashboard (`valor-time-titular.csv` é uma entrada manual).
- `pyproject.toml` / `uv.lock`: dependências do projeto.
- `Dockerfile` & `docker-compose.yml`: arquivos de configuração Docker.

## Autor
Desenvolvido por [Roberto Schneider](https://www.linkedin.com/in/robertoschneider/).
2025.
