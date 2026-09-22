FROM python:3.12-slim

# uv é usado para instalar as dependências a partir do pyproject.toml/uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

# o ambiente fica fora de /app para não ser sobrescrito pelo volume montado
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-group pipeline

COPY app ./app
COPY atualizar_previsoes.py ./

# Porta padrão do Streamlit
EXPOSE 8504

CMD ["/opt/venv/bin/streamlit", "run", "app/app.py", "--server.port=8504", "--server.address=0.0.0.0"]
