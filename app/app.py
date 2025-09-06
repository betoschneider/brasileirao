import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime, time


# Função para carregar os dados
def carregar_dados():
    df = pd.read_csv('tabela-evolucao-modelos.csv')
    df.rename(columns={'time': 'Time',
                       'posicao': 'Posição',
                       'pontos': 'Pontos',
                       'rodada_num': 'Rodada',
                       'modelo': 'Modelo',
                       }, inplace=True)
    df['data_atualizacao'] = pd.to_datetime(df['data_atualizacao'])
    df_partidas = pd.read_csv('partidas-modelos.csv')
    return df, df_partidas

# Interface Streamlit
def main():

    # Configurações da página
    st.set_page_config(
        page_title="Brasileirão Série A - Previsão de resultados",
        page_icon="⚽",
        layout="wide",
    )

    # Carregar dados
    df = carregar_dados()
    partidas = df[1]
    df = df[0]
    
    # sidebar
    st.sidebar.write(f"Dados atualizados em: {df['data_atualizacao'].max().strftime('%d/%m/%Y %H:%M')}.")
    st.sidebar.header("Filtros")

    # filtro de modelo
    modelos = np.sort(df['Modelo'].unique())
    modelo = st.sidebar.selectbox(
        "Modelo",
        modelos,
        index=2,
        # label_visibility="hidden",
    )
    
    # filtro de times
    times = np.sort(df['Time'].unique())
    col1, col2 = st.sidebar.columns([1, 4])
    with col1:
        st.subheader("Times  ")
    with col2:
        # Botão para restaurar checkboxes
        if st.button(":arrows_counterclockwise:", type="tertiary"):
            for t in times:
                st.session_state[f"time_{t}"] = False

    # Checkboxes com controle de estado
    times_selecionados = []
    for t in times:
        if st.sidebar.checkbox(t, value=False, key=f"time_{t}"):
            times_selecionados.append(t)
    # fim sidebar

    
    # área principal

    # Filtrar dados com base no modelo selecionado
    # dados do gráfico
    df = df[
        (df['Modelo'] == modelo)
        ]
    # dados da tabela
    tabela = df[(df['Rodada'] == 38)][['Posição', 'Time', 'Pontos']].sort_values('Posição')

    # Filtrar por times selecionados
    # dados do gráfico
    if times_selecionados:
        df = df[df['Time'].isin(times_selecionados)]

    # títulos da área principal
    # temporada
    temporada = df['data_atualizacao'].max().year
    st.title(f"Brasileirão Série A {temporada}")
    st.subheader(f"Previsão de resultados utilizando modelos de machine learning até a 38ª rodada. ")
    st.markdown(f"---")
    st.subheader(f"Evolução da posição na tabela")
    st.write(f"Modelo: {modelo}")

    # gráfico
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Rodada:Q", scale=alt.Scale(domain=[1, 38]), title="Rodada"),
            y=alt.Y("Posição:Q", scale=alt.Scale(domain=[20, 1]), title="Posição"),
            color="Time:N",
        )
        .properties(
            width=700,
            height=700,
        )
    )
    st.altair_chart(chart, use_container_width=True)

    # tabela
    # === Conteúdo centralizado abaixo ===
    col_esq, col_centro, col_dir = st.columns([1, 1, 1])
    with col_centro:
        if len(times_selecionados) == 1:
            partidas = partidas[(partidas['time'] == times_selecionados[0]) & ((partidas['modelo'] == modelo) | (partidas['status'] == 'Finalizado'))]
            st.subheader(f"Resultados previstos para {times_selecionados[0]}")
            st.dataframe(partidas, hide_index=True, height=740)
        st.subheader(f"Previsão de pontos na 38ª rodada")
        st.dataframe(tabela, hide_index=True, height=740)

    # modelos utilizados
    st.markdown("""
        ### Sobre os modelos de machine learning e técnicas de validação utilizados
                
        #### Modelos
        **Logistic Regression**
        - Modelo estatístico para classificação binária.
        - Estima a probabilidade de uma classe usando a função logística.
        - Fácil de interpretar e rápido de treinar.

        **Random Forest**
        - Ensemble de árvores de decisão.
        - Combina múltiplas árvores para reduzir overfitting.
        - Funciona bem com dados tabulares e variáveis categóricas/métricas.

        #### Validação
        **Leave-One-Season-Out (LOSO)**
        - Variante de cross-validation temporal.
        - Cada “temporada” do dataset é usada uma vez como teste, enquanto o restante serve como treino.
        - Útil em séries temporais esportivas ou financeiras.

        **Monte Carlo Cross-Validation (MCCV)**
        - Amostras aleatórias do dataset são divididas em treino e teste múltiplas vezes.
        - Resultados são agregados para estimar desempenho do modelo.
        - Mais flexível que k-fold, mas não garante que cada ponto de dado será usado como teste.
        """)
    
    # rodapé
    st.markdown(f"""
        ---
        Desenvolvido por [Roberto Schneider](https://www.linkedin.com/in/robertoschneider/) | [Repositório](https://github.com/betoschneider/brasileirao) | {datetime.now().year}.
        """)
 

if __name__ == '__main__':
    main()