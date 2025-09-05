# Usar imagem Python oficial
FROM python:3.12

# Diretório de trabalho
WORKDIR /app

# Copiar arquivos
COPY app ./app

# Instalar dependências
# RUN pip install --no-cache-dir streamlit pandas requests
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Porta padrão do Streamlit
EXPOSE 8504

# Comando para iniciar o app
CMD ["streamlit", "run", "app/app.py", "--server.port=8504", "--server.address=0.0.0.0"]