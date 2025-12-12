# Brasileirão Série A - Previsão de Resultados

## Sobre o Projeto
Este projeto utiliza técnicas de Ciência de Dados e Machine Learning para analisar e prever os resultados das partidas do Campeonato Brasileiro Série A. O sistema coleta dados, processa estatísticas e aplica diversos modelos preditivos para estimar a classificação final e os resultados rodada a rodada.

## Funcionalidades
- **Análise de Dados**: Processamento de dados históricos e atuais do campeonato.
- **Modelos de Machine Learning**: Utilização de algoritmos como Random Forest, Logistic Regression, Gradient Boosting, KNN, Naive Bayes e Voting Classifier.
- **Validação Robusta**: Técnicas de validação como Leave-One-Season-Out (LOSO) e Monte Carlo Cross-Validation (MCCV).
- **Interface Web (Streamlit)**:
    - Visualização da tabela prevista.
    - Gráficos de evolução dos times.
    - Previsões detalhadas de partidas futuras.
    - Estatísticas de desempenho.

## Tecnologias Utilizadas
- **Linguagem**: Python
- **Análise de Dados**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualização**: Altair, Streamlit
- **Containerização**: Docker

## Como Executar

### Pré-requisitos
- Python 3.8+
- Docker (opcional)

### Instalação Local
1. Clone o repositório:
   ```bash
   git clone https://github.com/betoschneider/brasileirao.git
   cd brasileirao
   ```
2. Crie um ambiente virtual e instale as dependências:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

### Executando a Análise
Para rodar os notebooks e atualizar os modelos:
```bash
jupyter notebook brasileirao-analise.ipynb
```

### Executando a Aplicação Web
Para iniciar a interface do Streamlit:
```bash
streamlit run app/app.py
```

### Usando Docker
1. Construa e inicie os containers:
   ```bash
   docker-compose up --build
   ```
2. Acesse a aplicação em `http://localhost:8501`.

## Estrutura do Projeto
- `brasileirao-analise.ipynb`: Notebook principal com a lógica de ETL e modelagem.
- `app/`: Diretório contendo a aplicação Streamlit e arquivos de dados CSV.
- `Dockerfile` & `docker-compose.yml`: Arquivos de configuração Docker.

## Autor
Desenvolvido por [Roberto Schneider](https://www.linkedin.com/in/robertoschneider/).
2025.
