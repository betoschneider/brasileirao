import requests
import urllib3
import pandas as pd
import streamlit as st

# Ignorar warnings de certificado SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="Brasileirão 2025", layout="wide")

# Função para buscar e processar dados
@st.cache_data
def carregar_dados():
    url = "https://service.ig.com.br/football_ig/campeonatos/10/fases/768"
    response = requests.get(url, verify=False)
    data = response.json()

    # Dados da edição
    edicao_data = {}
    if "edicao" in data:
        for k, v in data["edicao"].items():
            edicao_data[f"edicao.{k}"] = v

    # Partidas
    partidas_data = []
    if "partidas" in data:
        for rodada, lista_partidas in data["partidas"].items():
            for partida in lista_partidas:
                registro = {"rodada": rodada}
                registro.update(partida)
                registro.update(edicao_data)
                partidas_data.append(registro)

    partidas_df = pd.json_normalize(partidas_data)

    # Remove colunas indesejadas em partidas_df
    colunas_descartar_partidas = [
        "partida_id",
        "disputa_penalti",
        "slug",
        "data_realizacao",
        "hora_realizacao",
        "data_realizacao_iso",
        "_link",
        "edicao.nome",
        "edicao.slug",
        "estadio",
    ]
    partidas_df = partidas_df.drop(columns=[c for c in colunas_descartar_partidas if c in partidas_df.columns])

    # Tabela
    tabela_data = []
    if "tabela" in data:
        for posicao in data["tabela"]:
            registro = dict(posicao)
            registro.update(edicao_data)
            tabela_data.append(registro)

    tabela_df = pd.json_normalize(tabela_data)

    # Remove coluna variacao_posicao
    if "variacao_posicao" in tabela_df.columns:
        tabela_df = tabela_df.drop(columns=["variacao_posicao"])

    # Calcula aproveitamento recente
    def calc_aproveitamento(ultimos):
        if not ultimos:
            return None
        pontos = sum(3 if r == "v" else 1 if r == "e" else 0 for r in ultimos)
        return round(pontos / 15, 2) * 100 if len(ultimos) >= 5 else None

    if "ultimos_jogos" in tabela_df.columns:
        tabela_df["aproveitamento_recente"] = tabela_df["ultimos_jogos"].apply(calc_aproveitamento)

    return partidas_df, tabela_df

# Carrega os dados
partidas_df, tabela_df = carregar_dados()

# Título
st.title("📊 Campeonato Brasileiro 2025")

# Tabela de classificação
st.subheader("Classificação")
st.dataframe(tabela_df, use_container_width=True)

csv_tabela = tabela_df.to_csv(index=False).encode('utf-8')
st.download_button("Baixar classificação (CSV)", csv_tabela, "tabela.csv", "text/csv")

# Tabela de partidas
st.subheader("Partidas")
st.dataframe(partidas_df, use_container_width=True)

csv_partidas = partidas_df.to_csv(index=False).encode('utf-8')
st.download_button("Baixar partidas (CSV)", csv_partidas, "partidas.csv", "text/csv")
