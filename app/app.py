import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime


# Função para carregar os dados
def carregar_dados():
    # tabela acumulada
    df = pd.read_csv('tabela-evolucao-modelos.csv')
    df.rename(columns={'time': 'Time',
                       'posicao': 'Posição',
                       'pontos': 'Pontos',
                       'vitorias': 'Vitórias',
                       'rodada_num': 'Rodada',
                       'modelo': 'Modelo',
                       }, inplace=True)
    df['data_atualizacao'] = pd.to_datetime(df['data_atualizacao'])
    df['Aproveitamento Previsto'] = df['Pontos'] / (df['Rodada'] * 3)
    # df['Tendência (%)'] = (df['Aproveitamento Previsto'] / df['aproveitamento']) - 1
    df['Tendência'] = (df['Aproveitamento Previsto'] - df['aproveitamento']) 
    def formatar_tendencia(valor):
        if pd.isna(valor):
            return "-"
        elif valor > 0:
            return f"🟢 +{valor*100:.2f} p.p."
        elif valor < 0:
            return f"🔴 {valor*100:.2f} p.p."
        else:
            return f"⚪ {valor*100:.2f} p.p."

    df["Tendência"] = df["Tendência"].apply(formatar_tendencia)
    df.rename(columns={
        'aproveitamento': 'Aproveitamento Atual', 
        'Aproveitamento Previsto': 'Aproveitamento'
        }, inplace=True)
    df['Aproveitamento'] = df['Aproveitamento'].apply(lambda x: f"{x:.2%}")

    # partidas
    df_partidas = pd.read_csv('partidas-modelos.csv')
    df_partidas.rename(columns={'rodada_num': 'Rodada',
                                'status': 'Status',
                                'mandante': 'Mandante',
                                'time': 'Time',
                                'adversario': 'Adversário',
                                'resultado': 'Resultado',
                                'frequencia': 'Frequência',
                                'modelo': 'Modelo',
                               }, inplace=True)
    df_partidas['Frequência'] = df_partidas['Frequência'].round(2)
    df_partidas['Frequência'] = df_partidas['Frequência'].fillna('-')

    # estatisticas dos times
    df_times_stats = pd.read_csv('times-stats.csv')
    df_times_stats.rename(columns={'time': 'Time',
                                   'pontos_time': 'Pontos',
                                   'jogos': 'Jogos',
                                   'aproveitamento': 'Aproveitamento',
                                   'aproveitamento_recente': 'Aproveitamento Recente',
                                   'media_gols_marcados': 'Média Gols Marcados',
                                   'media_gols_sofridos': 'Média Gols Sofridos'
                                  }, inplace=True)
    return df, df_partidas, df_times_stats


# Interface Streamlit
def main():

    # Configurações da página
    st.set_page_config(
        page_title="Brasileirão Série A - Previsão de resultados",
        page_icon="⚽",
        # layout="wide",
    )

    # Carregar dados
    df, partidas, stats = carregar_dados()

    # título e subtítulo
    dt_atualizacao = df['data_atualizacao'].max()
    temporada = dt_atualizacao.year
    st.title(f"Brasileirão Série A {temporada}")
    st.subheader("Projeção de resultados com machine learning até a 38ª rodada.")
    st.write(f"Última atualização: {dt_atualizacao.strftime('%d/%m/%Y %H:%M')}")
    st.markdown("---")

         

    # ================================
    # 1️⃣ Gráfico de barras pontos finais no topo
    # ================================
    tabela = df[df['Rodada'] == 38][['Posição', 'Time', 'Pontos', 'Vitórias', 'Aproveitamento', 'Tendência']].sort_values(["Pontos", 'Vitórias'], ascending=False).reset_index(drop=True)
    tabela["Ranking"] = tabela.index + 1

    def classificar_cor(rank):
        if rank == 1: return "Campeão"
        elif 2 <= rank <= 4: return "G4"
        elif 5 <= rank <= 6: return "G6"
        elif 7 <= rank <= 10: return "Primeira metade"
        elif 17 <= rank <= 20: return "Rebaixado"
        else: return "Segunda metade"

    tabela["Grupo"] = tabela["Ranking"].apply(classificar_cor)
    cores = {
        "Campeão": "#1f77b4",
        "G4": "#2ca02c",
        "G6": "#98df8a",
        "Primeira metade": "#BABCBE",
        "Segunda metade": "#BABCBE",
        "Rebaixado": "#d62728"
    }

    bars = (
        alt.Chart(tabela)
        .mark_bar()
        .encode(
            x=alt.X("Pontos:Q", axis=None),
            y=alt.Y("Time:N", sort="-x", title=None),
            color=alt.Color("Grupo:N", scale=alt.Scale(domain=list(cores.keys()), range=list(cores.values())), legend=None),
            tooltip=["Time:N", "Pontos:Q", "Posição:O", "Grupo:N"]
        )
        .properties(height=500)
    )

    labels = (
        alt.Chart(tabela)
        .mark_text(align="left", baseline="middle", dx=3, color="black")
        .encode(x="Pontos:Q", y=alt.Y("Time:N", sort="-x"), text="Pontos:Q")
    )

    st.subheader("Pontuação prevista ao fim do campeonato")
    tab1, tab2 = st.tabs(["📑 Tabela", "📊 Gráfico"])  
    with tab1:
        st.dataframe(tabela[['Posição', 'Time', 'Pontos', 'Vitórias', 'Aproveitamento', 'Tendência']], hide_index=True, height=740)


    with tab2: 
        st.altair_chart(bars + labels, use_container_width=True)

    st.markdown("---")

    # ================================
    # Seleção de times
    # ================================
    times = np.sort(df['Time'].unique())
    times_selecionados = st.multiselect("Selecione times para ver as rodadas previstas:", times, placeholder="Todos os times")

    if len(times_selecionados)==1:
        stats = stats[stats['Time'].isin(times_selecionados)]
        jogos = stats['Jogos'].mean()
        pontos = stats['Pontos'].mean()
        aproveitamento = stats['Aproveitamento'].mean()
        aproveitamento_recente= stats['Aproveitamento Recente'].mean()
        gols_marcados = stats['Média Gols Marcados'].mean()
        gols_sofridos = stats['Média Gols Sofridos'].mean()

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        
        col1.metric('Pontos', f'{pontos:.0f}')
        col2.metric('Jogos', f'{jogos:.0f}')
        col3.metric('Aproveitamento', f'{aproveitamento:.2%}')
        col4.metric('Aproveitamento Recente', f'{aproveitamento_recente:.2%}')
        col5.metric('Gols marcados', f'{gols_marcados:.2f}')
        col6.metric('Gols sofridos', f'{gols_sofridos:.2f}')
        st.markdown("---")

    # ================================
    # 2️⃣ Evolução rodada a rodada
    # ================================
    if times_selecionados:
        df = df[df['Time'].isin(times_selecionados)]

    qtd_times = df['Time'].nunique()
    
    if qtd_times == 1:
        st.subheader(f"Posição na tabela a cada rodada - {times_selecionados[0]}")
    else:
        st.subheader(f"Posição na tabela a cada rodada - {qtd_times} times selecionados")

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Rodada:Q", scale=alt.Scale(domain=[1, 38]), title="Rodada"),
            y=alt.Y("Posição:Q", scale=alt.Scale(domain=[20, 1]), title="Posição"),
            color="Time:N",
        )
        .properties(height=700)
    )
    st.altair_chart(chart, use_container_width=True)

    # ================================
    # Identificação do modelo
    # ================================
    modelo_usado = df['Modelo'].mode()[0] if not df['Modelo'].mode().empty else "Desconhecido"
    chave_modelo = modelo_usado.split('_')[0]
    chave_validacao = modelo_usado.split('_')[-1]

    # ================================
    # Seleção de rodadas
    # ================================
    rodadas = np.sort(partidas['Rodada'].unique())
    rodadas_selecionadas = st.multiselect("", rodadas, placeholder="Todas as rodadas")

    # ================================
    # 3️⃣ Tabela de jogos futuros
    # ================================
    if times_selecionados:
        partidas_time = partidas[(partidas['Time'].isin(times_selecionados)) & (partidas['Status'] != 'Finalizado')]
        colunas = ['Rodada', 'Status', 'Mandante', 'Adversário', 'Resultado', 'Frequência']
    else:
        partidas_time = partidas[partidas['Status'] != 'Finalizado']
        colunas = ['Time', 'Rodada', 'Status', 'Mandante', 'Adversário', 'Resultado', 'Frequência']
        
    partidas_time = partidas_time[colunas]

    if rodadas_selecionadas:
        partidas_time = partidas_time[partidas_time['Rodada'].isin(rodadas_selecionadas)]

    if chave_validacao != "MCCV":
        partidas_time = partidas_time.drop(columns=['Frequência'])
    
    partidas_time['Rodada'] = partidas_time['Rodada'].astype(str)
    if qtd_times == 1:
        st.subheader(f"Previsões para os jogos do {times_selecionados[0]}")
    else:
        st.subheader(f"Previsões para os jogos - {qtd_times} times selecionados")
    
    resultados = partidas_time['Resultado'].value_counts().to_dict()
    st.markdown(f"**✌️ Vitórias:** {resultados.get('Vitória', 0)} | **🤝 Empates:** {resultados.get('Empate', 0)} | **👎Derrotas:** {resultados.get('Derrota', 0)}")
    st.dataframe(partidas_time, hide_index=True, height=min(600, 38*partidas_time.shape[0]))

    # ================================
    # 4️⃣ Descrição do modelo e validação
    # ================================
    det_modelos = {
        "LogisticRegression": {
            "nome": "#### Logistic Regression",
            "descricao": "- Modelo estatístico para classificação.\n- Estima probabilidades com a função logística.\n- Fácil de interpretar e rápido."
        },
        "RandomForest": {
            "nome": "#### Random Forest",
            "descricao": "- Ensemble de múltiplas árvores de decisão.\n- Reduz overfitting.\n- Funciona bem com dados tabulares."
        },
        "GradientBoosting": {
            "nome": "#### Gradient Boosting",
            "descricao": "- Ensemble sequencial de árvores.\n- Cada árvore corrige erros das anteriores.\n- Alta acurácia, mas mais lento."
        },
        "KNN": {
            "nome": "#### K-Nearest Neighbors (KNN)",
            "descricao": "- Classifica com base nos vizinhos mais próximos.\n- Simples e intuitivo.\n- Pode ser lento em bases grandes."
        },
        "NaiveBayes": {
            "nome": "#### Naive Bayes",
            "descricao": "- Baseado no Teorema de Bayes.\n- Assume independência entre features.\n- Muito rápido e eficiente em texto."
        }
    }

    det_validacao = {
        "LOSO": {
            "nome": "#### Leave-One-Season-Out (LOSO)",
            "descricao": "- Cada temporada é teste uma vez.\n- O restante é treino.\n- Bom para séries temporais esportivas."
        },
        "MCCV": {
            "nome": "#### Monte Carlo Cross-Validation (MCCV)",
            "descricao": "- Amostras aleatórias divididas em treino e teste várias vezes.\n- Resultados agregados estimam o desempenho.\n- Mais flexível que k-fold."
        }
    }

    st.markdown("### Sobre os modelos e técnicas de validação")
    st.markdown(det_modelos[chave_modelo]["nome"])
    st.markdown(det_modelos[chave_modelo]["descricao"])
    st.markdown(det_validacao[chave_validacao]["nome"])
    st.markdown(det_validacao[chave_validacao]["descricao"])

    # ================================
    # Rodapé
    # ================================
    st.markdown(f"""
        ---
        Desenvolvido por [Roberto Schneider](https://www.linkedin.com/in/robertoschneider/) | 
        [Repositório](https://github.com/betoschneider/brasileirao) | {datetime.now().year}.
        """)


if __name__ == '__main__':
    main()
