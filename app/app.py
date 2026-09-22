"""Dashboard Streamlit com as previsões do Brasileirão Série A.

Lê os arquivos gerados por ``atualizar_previsoes.py`` e apresenta a tabela
projetada, a evolução rodada a rodada e as previsões de cada partida.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DIR_DADOS = Path(__file__).resolve().parent
RAIZ = DIR_DADOS.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

try:
    from atualizar_previsoes import EXECUTAR_LOSO, EXECUTAR_MCCV
except ImportError:
    EXECUTAR_LOSO = False
    EXECUTAR_MCCV = True
ARQUIVO_EVOLUCAO = DIR_DADOS / "tabela-evolucao-modelos.csv"
ARQUIVO_PARTIDAS = DIR_DADOS / "partidas-modelos.csv"
ARQUIVO_TIMES_STATS = DIR_DADOS / "times-stats.csv"
ARQUIVO_CHANCES = DIR_DADOS / "chances-times.csv"

# aproveitamento é sempre armazenado como fração (0 a 1) e exibido como %
CORES = {
    "Campeão": "#1f77b4",
    "G4": "#2ca02c",
    "G6": "#98df8a",
    "Primeira metade": "#BABCBE",
    "Segunda metade": "#BABCBE",
    "Rebaixado": "#d62728",
}

DESCRICAO_MODELOS = {
    "LogisticRegression": (
        "Regressão logística",
        (
            "Modelo estatístico de classificação que estima a probabilidade de cada "
            "resultado (vitória, empate ou derrota) a partir das variáveis de entrada."
        ),
    ),
    "RandomForest": (
        "Random Forest",
        (
            "Conjunto de árvores de decisão treinadas em amostras diferentes. "
            "Reduz o overfitting e lida bem com dados tabulares."
        ),
    ),
    "GradientBoosting": (
        "Gradient Boosting",
        (
            "Árvores treinadas de forma sequencial, cada uma corrigindo os erros da "
            "anterior. Costuma ter alta acurácia, mas é mais lento para treinar."
        ),
    ),
    "KNN": (
        "K-Nearest Neighbors (KNN)",
        (
            "Classifica cada partida a partir das partidas mais parecidas do "
            "histórico. Simples e intuitivo, porém sensível à escala das variáveis."
        ),
    ),
    "VotingClassifier": (
        "Voting Classifier",
        (
            "Combina as probabilidades de Random Forest, Regressão Logística e "
            "Gradient Boosting, reduzindo o viés de cada modelo isolado."
        ),
    ),
}

DESCRICAO_VALIDACAO = {
    "LOSO": (
        "Leave-One-Season-Out (LOSO)",
        (
            "Treina com as temporadas anteriores e testa na temporada seguinte. "
            "Mede o desempenho do modelo em um campeonato que ele nunca viu."
        ),
    ),
    "MCCV": (
        "Monte Carlo Cross-Validation (MCCV)",
        (
            "Repete centenas de divisões aleatórias entre treino e teste e projeta a "
            "partida em cada uma delas. A frequência das previsões vira a confiança "
            "de cada resultado."
        ),
    ),
}


def carregar_modelos() -> list[str]:
    """Lista os modelos disponíveis no arquivo de evolução."""
    modelos = list(pd.read_csv(ARQUIVO_EVOLUCAO)["modelo"].dropna().unique())
    if not EXECUTAR_LOSO:
        modelos = [m for m in modelos if not m.endswith("_LOSO")]
    if not EXECUTAR_MCCV:
        modelos = [m for m in modelos if not m.endswith("_MCCV")]
    return sorted(modelos)


def calcular_status_matematico(
    stats_df: pd.DataFrame, rodada_final: int = 38
) -> dict[str, dict[str, str]]:
    """Calcula a garantia ou eliminação matemática para título, G4 e rebaixamento."""
    df = stats_df.copy()
    col_jogos = "Jogos Disputados" if "Jogos Disputados" in df.columns else "jogos"
    col_pontos = "Pontos" if "Pontos" in df.columns else "pontos_time"
    col_time = "Time" if "Time" in df.columns else "time"

    df = df.set_index(col_time)
    jogos_restantes = (rodada_final - df[col_jogos]).clip(lower=0)
    p_atual = df[col_pontos].astype(float)
    p_max = p_atual + 3.0 * jogos_restantes

    status = {}
    times = list(df.index)
    for t in times:
        outros = [outro for outro in times if outro != t]
        outros_max_desc = sorted([p_max[outro] for outro in outros], reverse=True)
        outros_atual_desc = sorted([p_atual[outro] for outro in outros], reverse=True)

        # Título (1º lugar)
        if p_atual[t] > outros_max_desc[0]:
            st_titulo = "garantido"
        elif p_max[t] < outros_atual_desc[0]:
            st_titulo = "eliminado"
        else:
            st_titulo = "aberto"

        # G4 (1º ao 4º lugar)
        if p_atual[t] > outros_max_desc[3]:
            st_g4 = "garantido"
        elif p_max[t] < outros_atual_desc[3]:
            st_g4 = "eliminado"
        else:
            st_g4 = "aberto"

        # Rebaixamento (17º ao 20º lugar)
        if p_max[t] < outros_atual_desc[15]:
            st_z4 = "rebaixado"
        elif p_atual[t] > outros_max_desc[15]:
            st_z4 = "salvo"
        else:
            st_z4 = "aberto"

        status[t] = {"titulo": st_titulo, "g4": st_g4, "rebaixamento": st_z4}
    return status


def carregar_dados(modelo: str):
    """Carrega e normaliza os arquivos consumidos pelo dashboard."""
    df = pd.read_csv(ARQUIVO_EVOLUCAO)
    df = df[df["modelo"] == modelo].copy()
    df = df.rename(
        columns={
            "time": "Time",
            "rodada_num": "Rodada",
            "pontos": "Pontos",
            "vitorias": "Vitórias",
            "posicao": "Posição",
            "modelo": "Modelo",
            "pontos_time": "Pontos Atuais",
            "aproveitamento": "Aproveitamento Atual",
            "aproveitamento_recente": "Aproveitamento Recente",
            "media_gols_marcados": "Média de Gols Marcados",
            "media_gols_sofridos": "Média de Gols Sofridos",
            "jogos": "Jogos Disputados",
        }
    )
    df["data_atualizacao"] = pd.to_datetime(df["data_atualizacao"])
    # a rodada é inteira; o CSV pode trazê-la como float após os merges
    df["Rodada"] = df["Rodada"].astype(int)

    # aproveitamento previsto (fração) e a variação em relação ao aproveitamento atual
    df["Aproveitamento Previsto"] = df["Pontos"] / (df["Rodada"] * 3)
    df["Variação de Aproveitamento"] = (
        df["Aproveitamento Previsto"] - df["Aproveitamento Atual"]
    )

    partidas = pd.read_csv(ARQUIVO_PARTIDAS)
    partidas = partidas[partidas["modelo"] == modelo].copy()
    partidas = partidas.rename(
        columns={
            "rodada_num": "Rodada",
            "status": "Status",
            "mandante": "Mandante",
            "time": "Time",
            "adversario": "Adversário",
            "resultado": "Resultado",
            "frequencia": "Confiança",
            "modelo": "Modelo",
        }
    )
    partidas["Rodada"] = partidas["Rodada"].astype(int)

    stats = pd.read_csv(ARQUIVO_TIMES_STATS).rename(
        columns={
            "time": "Time",
            "pontos_time": "Pontos",
            "jogos": "Jogos Disputados",
            "aproveitamento": "Aproveitamento",
            "aproveitamento_recente": "Aproveitamento Recente",
            "media_gols_marcados": "Média de Gols Marcados",
            "media_gols_sofridos": "Média de Gols Sofridos",
        }
    )
    stats["Jogos Disputados"] = stats["Jogos Disputados"].astype(int)

    chances = carregar_chances(modelo)
    if chances is not None:
        status_mat = calcular_status_matematico(stats)
        if "status_titulo" not in chances.columns:
            chances["status_titulo"] = chances["Time"].map(
                lambda t: status_mat.get(t, {}).get("titulo", "aberto")
            )
        if "status_g4" not in chances.columns:
            chances["status_g4"] = chances["Time"].map(
                lambda t: status_mat.get(t, {}).get("g4", "aberto")
            )
        if "status_rebaixamento" not in chances.columns:
            chances["status_rebaixamento"] = chances["Time"].map(
                lambda t: status_mat.get(t, {}).get("rebaixamento", "aberto")
            )
    return df, partidas, stats, chances


def carregar_chances(modelo: str) -> pd.DataFrame | None:
    """Carrega as probabilidades do modelo selecionado, quando existirem.

    As chances vêm das simulações Monte Carlo, então só existem para o modelo
    validado dessa forma.
    """
    if not ARQUIVO_CHANCES.exists():
        return None

    chances = pd.read_csv(ARQUIVO_CHANCES)
    chances = chances[chances["modelo"] == modelo].copy()
    if chances.empty:
        return None

    return chances.rename(
        columns={
            "time": "Time",
            "pontos_previstos": "Pontos Previstos",
            "chance_pontos_previstos": "Chance da Pontuação",
            "pontos_p10": "Piso (p10)",
            "pontos_p90": "Teto (p90)",
            "chance_titulo": "Chance de Título",
            "chance_g4": "Chance de G4",
            "chance_rebaixamento": "Chance de Rebaixamento",
        }
    )


def classificar_grupo(posicao: int) -> str:
    if posicao == 1:
        return "Campeão"
    if 2 <= posicao <= 4:
        return "G4"
    if 5 <= posicao <= 6:
        return "G6"
    if 7 <= posicao <= 10:
        return "Primeira metade"
    if 17 <= posicao <= 20:
        return "Rebaixado"
    return "Segunda metade"


def formatar_variacao(valor: float) -> str:
    """Formata a variação de aproveitamento em pontos percentuais."""
    if pd.isna(valor):
        return "-"
    if valor > 0:
        return f"🔺 +{valor * 100:.2f} p.p."
    if valor < 0:
        return f"🔻 {valor * 100:.2f} p.p."
    return f"◽ {valor * 100:.2f} p.p."


def formatar_chance(valor: float, status: str = "aberto") -> str:
    """Formata uma probabilidade levando em consideração a situação matemática.

    - Se status == 'garantido' ou 'rebaixado': '100%'
    - Se status == 'eliminado' ou 'salvo': '—'
    - Se status == 'aberto':
        - Se valor >= 0.995: '>99%' (não crava 100% enquanto houver chance matemática)
        - Se valor <= 0 ou pd.isna(valor): '—'
        - Se valor < 0.01: '<1%'
        - Caso contrário: f'{valor:.0%}'
    """
    if pd.isna(valor):
        return "—"
    if status in ("garantido", "rebaixado"):
        return "100%"
    if status in ("eliminado", "salvo"):
        return "—"
    if valor >= 0.995:
        return ">99%"
    if valor <= 0:
        return "—"
    if valor < 0.01:
        return "<1%"
    return f"{valor:.0%}"


def montar_tabela_final(
    df: pd.DataFrame, rodada_final: int, chances: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Monta a tabela projetada para a última rodada, já ordenada e classificada."""
    colunas = [
        "Posição",
        "Time",
        "Pontos",
        "Vitórias",
        "Aproveitamento Previsto",
        "Aproveitamento Atual",
        "Variação de Aproveitamento",
    ]
    tabela = (
        df[df["Rodada"] == rodada_final][colunas]
        .sort_values(["Pontos", "Vitórias"], ascending=False)
        .reset_index(drop=True)
    )
    tabela["Grupo"] = tabela["Posição"].astype(int).apply(classificar_grupo)

    if chances is not None:
        tabela = tabela.merge(
            chances[["Time", "Chance da Pontuação", "Piso (p10)", "Teto (p90)"]],
            on="Time",
            how="left",
        )
        tabela["Faixa Provável"] = tabela.apply(
            lambda linha: f"{linha['Piso (p10)']:.0f} – {linha['Teto (p90)']:.0f}",
            axis=1,
        )
    return tabela


def grafico_chance(
    chances: pd.DataFrame, coluna: str, titulo: str, coluna_status: str | None = None
) -> alt.Chart | None:
    """Barras horizontais com os times mais prováveis em um cenário."""
    dados = (
        chances[chances[coluna] > 0].sort_values(coluna, ascending=False).head(8).copy()
    )
    if dados.empty:
        return None
    dados["Chance_Exibicao"] = dados.apply(
        lambda r: formatar_chance(
            r[coluna], r.get(coluna_status, "aberto") if coluna_status else "aberto"
        ),
        axis=1,
    )
    return (
        alt.Chart(dados, title=titulo)
        .mark_bar(color="#6b7280")
        .encode(
            x=alt.X(
                f"{coluna}:Q",
                title="Chance",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("Time:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Time:N"),
                alt.Tooltip("Chance_Exibicao:N", title="Chance"),
            ],
        )
        .properties(height=260)
    )


def renderizar_chances(chances: pd.DataFrame) -> None:
    """Mostra as probabilidades calculadas pelas simulações Monte Carlo."""
    st.caption(
        "Probabilidades estimadas pelas simulações do modelo. A **chance da pontuação** é a "
        "frequência com que o time terminou exatamente com os pontos previstos e a **faixa "
        "provável** cobre de 10% a 90% dos cenários. Como o campeonato segue em andamento, "
        "probabilidades extremas são exibidas como **>99%** ou **<1%**, reservando **100%** e "
        "**0%** exclusivamente para confirmações matemáticas oficiais."
    )

    tabela = (
        chances[
            [
                "Time",
                "Pontos Previstos",
                "Chance da Pontuação",
                "Piso (p10)",
                "Teto (p90)",
                "Chance de Título",
                "Chance de G4",
                "Chance de Rebaixamento",
                "status_titulo",
                "status_g4",
                "status_rebaixamento",
            ]
        ]
        .sort_values(
            ["Chance de Título", "Chance de G4", "Pontos Previstos"], ascending=False
        )
        .reset_index(drop=True)
    )
    tabela["Faixa Provável"] = tabela.apply(
        lambda linha: f"{linha['Piso (p10)']:.0f} – {linha['Teto (p90)']:.0f}", axis=1
    )
    tabela["Chance da Pontuação"] = tabela["Chance da Pontuação"].apply(formatar_chance)
    tabela["Chance de Título"] = tabela.apply(
        lambda r: formatar_chance(r["Chance de Título"], r.get("status_titulo", "aberto")),
        axis=1,
    )
    tabela["Chance de G4"] = tabela.apply(
        lambda r: formatar_chance(r["Chance de G4"], r.get("status_g4", "aberto")),
        axis=1,
    )
    tabela["Chance de Rebaixamento"] = tabela.apply(
        lambda r: formatar_chance(
            r["Chance de Rebaixamento"], r.get("status_rebaixamento", "aberto")
        ),
        axis=1,
    )

    st.dataframe(
        tabela[
            [
                "Time",
                "Pontos Previstos",
                "Chance da Pontuação",
                "Faixa Provável",
                "Chance de Título",
                "Chance de G4",
                "Chance de Rebaixamento",
            ]
        ],
        hide_index=True,
        height=560,
    )

    grafico_titulo = grafico_chance(
        chances, "Chance de Título", "🏆 Corrida pelo título", "status_titulo"
    )
    grafico_rebaixamento = grafico_chance(
        chances, "Chance de Rebaixamento", "⚠️ Risco de rebaixamento", "status_rebaixamento"
    )

    coluna_esquerda, coluna_direita = st.columns(2)
    with coluna_esquerda:
        if grafico_titulo is not None:
            st.altair_chart(grafico_titulo, use_container_width=True)
    with coluna_direita:
        if grafico_rebaixamento is not None:
            st.altair_chart(grafico_rebaixamento, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Brasileirão Série A - Previsão de resultados",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    modelos = carregar_modelos()
    st.sidebar.title("⚙️ Configurações")
    st.sidebar.caption("Cada modelo é validado de uma forma diferente.")
    modelo = st.sidebar.selectbox("Modelo e validação", modelos)

    df, partidas, stats, chances = carregar_dados(modelo)
    chave_modelo = modelo.split("_")[0]
    chave_validacao = modelo.split("_")[-1]

    dt_atualizacao = df["data_atualizacao"].max()
    rodada_final = int(df["Rodada"].max())

    st.title(f"⚽ Brasileirão Série A {dt_atualizacao.year}")
    st.caption(
        f"Projeção de resultados com machine learning até a {rodada_final}ª rodada · "
        f"Última atualização em {dt_atualizacao.strftime('%d/%m/%Y às %H:%M')} · "
        f"Modelo **{modelo}**"
    )

    # ================================
    # 1) Pontuação prevista no fim do campeonato
    # ================================
    tabela = montar_tabela_final(df, rodada_final, chances)

    st.subheader("Classificação final prevista")

    barras = (
        alt.Chart(tabela)
        .mark_bar()
        .encode(
            x=alt.X("Pontos:Q", axis=None),
            y=alt.Y("Time:N", sort="-x", title=None),
            color=alt.Color(
                "Grupo:N",
                scale=alt.Scale(domain=list(CORES), range=list(CORES.values())),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Time:N"),
                alt.Tooltip("Posição:O", title="Posição"),
                alt.Tooltip("Pontos:Q", title="Pontos"),
                alt.Tooltip("Grupo:N"),
            ],
        )
        .properties(height=560)
    )
    rotulos = (
        alt.Chart(tabela)
        .mark_text(align="left", baseline="middle", dx=4, color="black")
        .encode(x="Pontos:Q", y=alt.Y("Time:N", sort="-x"), text="Pontos:Q")
    )

    def colorir_posicao(coluna: pd.Series) -> list[str]:
        return [
            f"background-color: {CORES[tabela.loc[indice, 'Grupo']]}; color: white; font-weight: bold;"
            for indice in coluna.index
        ]

    exibicao = tabela[
        [
            "Posição",
            "Time",
            "Pontos",
            "Vitórias",
            "Aproveitamento Previsto",
            "Aproveitamento Atual",
            "Variação de Aproveitamento",
        ]
    ].copy()
    if chances is not None:
        exibicao["Chance da Pontuação"] = tabela["Chance da Pontuação"].apply(
            formatar_chance
        )
        exibicao["Faixa Provável"] = tabela["Faixa Provável"]
    exibicao["Aproveitamento Previsto"] = exibicao["Aproveitamento Previsto"].map(
        "{:.2%}".format
    )
    exibicao["Aproveitamento Atual"] = exibicao["Aproveitamento Atual"].map(
        "{:.2%}".format
    )
    exibicao["Variação de Aproveitamento"] = exibicao[
        "Variação de Aproveitamento"
    ].apply(formatar_variacao)
    estilo = exibicao.style.apply(colorir_posicao, subset=["Posição"])

    rotulos_abas = ["📑 Tabela", "📊 Gráfico"]
    if chances is not None:
        rotulos_abas.append("🎲 Chances")
    abas = dict(zip(rotulos_abas, st.tabs(rotulos_abas)))

    with abas["📑 Tabela"]:
        st.dataframe(estilo, hide_index=True, height=720)
    with abas["📊 Gráfico"]:
        st.altair_chart(barras + rotulos, use_container_width=True)
    if chances is not None:
        with abas["🎲 Chances"]:
            renderizar_chances(chances)

    # ================================
    # 2) Seleção de times
    # ================================
    st.divider()
    times = np.sort(df["Time"].unique())
    times_selecionados = st.multiselect(
        "Filtrar por time (opcional)",
        times,
        placeholder="Todos os times",
    )

    if len(times_selecionados) == 1:
        selecionado = stats[stats["Time"].isin(times_selecionados)].iloc[0]
        colunas = st.columns(6)
        colunas[0].metric("Pontos conquistados", f"{selecionado['Pontos']:.0f}")
        colunas[1].metric("Jogos disputados", f"{selecionado['Jogos Disputados']:.0f}")
        colunas[2].metric("Aproveitamento", f"{selecionado['Aproveitamento']:.2%}")
        colunas[3].metric(
            "Aproveitamento recente", f"{selecionado['Aproveitamento Recente']:.2%}"
        )
        colunas[4].metric(
            "Média de gols marcados", f"{selecionado['Média de Gols Marcados']:.2f}"
        )
        colunas[5].metric(
            "Média de gols sofridos", f"{selecionado['Média de Gols Sofridos']:.2f}"
        )

    # ================================
    # 3) Evolução rodada a rodada
    # ================================
    if times_selecionados:
        df = df[df["Time"].isin(times_selecionados)]

    qtd_times = df["Time"].nunique()
    if qtd_times == 1:
        st.subheader(f"Posição na tabela a cada rodada · {times_selecionados[0]}")
    else:
        st.subheader(f"Posição na tabela a cada rodada · {qtd_times} times")

    grafico_evolucao = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Rodada:Q", scale=alt.Scale(domain=[1, rodada_final]), title="Rodada"
            ),
            y=alt.Y("Posição:Q", scale=alt.Scale(domain=[20, 1]), title="Posição"),
            color=alt.Color("Time:N", title=None),
            tooltip=[
                alt.Tooltip("Time:N"),
                alt.Tooltip("Rodada:Q", title="Rodada"),
                alt.Tooltip("Posição:Q", title="Posição"),
                alt.Tooltip("Pontos:Q", title="Pontos"),
            ],
        )
        .properties(height=640)
    )
    st.altair_chart(grafico_evolucao, use_container_width=True)

    # ================================
    # 4) Previsões das partidas
    # ================================
    st.divider()
    rodadas = np.sort(partidas["Rodada"].unique())
    rodadas_selecionadas = st.multiselect(
        "Filtrar por rodada (opcional)", rodadas, placeholder="Todas as rodadas"
    )

    if times_selecionados:
        partidas_time = partidas[partidas["Time"].isin(times_selecionados)]
        colunas_partidas = ["Rodada", "Status", "Mandante", "Adversário", "Resultado"]
        contexto = f"dos jogos de {', '.join(times_selecionados)}"
    else:
        partidas_time = partidas
        colunas_partidas = [
            "Rodada",
            "Time",
            "Status",
            "Mandante",
            "Adversário",
            "Resultado",
        ]
        contexto = "de todos os times"

    if chave_validacao == "MCCV":
        colunas_partidas = colunas_partidas + ["Confiança"]

    if rodadas_selecionadas:
        partidas_time = partidas_time[
            partidas_time["Rodada"].isin(rodadas_selecionadas)
        ]

    st.subheader(f"Previsões {contexto}")

    pendentes = partidas_time[partidas_time["Status"] != "Finalizado"]
    if not pendentes.empty:
        resultados = pendentes["Resultado"].value_counts().to_dict()
        st.markdown(
            f"**✌️ Vitórias:** {resultados.get('Vitória', 0)} · "
            f"**🤝 Empates:** {resultados.get('Empate', 0)} · "
            f"**👎 Derrotas:** {resultados.get('Derrota', 0)}"
        )
    else:
        st.caption("Nenhuma partida prevista para os filtros selecionados.")

    tabela_partidas = partidas_time[colunas_partidas].copy()
    if "Confiança" in tabela_partidas.columns:
        tabela_partidas["Confiança"] = tabela_partidas["Confiança"].apply(
            lambda valor: f"{valor:.0f}%" if pd.notna(valor) else "-"
        )
    st.dataframe(tabela_partidas, hide_index=True, height=560)
    if chave_validacao == "MCCV":
        st.caption(
            "A confiança é o percentual das simulações em que o resultado previsto "
            "apareceu."
        )

    # ================================
    # 5) Descrição do modelo e da validação
    # ================================
    st.divider()
    with st.expander("ℹ️ Como o modelo e a validação funcionam", expanded=False):
        nome_modelo, descricao_modelo = DESCRICAO_MODELOS[chave_modelo]
        nome_validacao, descricao_validacao = DESCRICAO_VALIDACAO[chave_validacao]
        st.markdown(f"**Modelo: {nome_modelo}**")
        st.write(descricao_modelo)
        st.markdown(f"**Validação: {nome_validacao}**")
        st.write(descricao_validacao)

    st.divider()
    st.caption(
        f"Desenvolvido por [Roberto Schneider](https://www.linkedin.com/in/robertoschneider/) · "
        f"[Repositório](https://github.com/betoschneider/brasileirao) · "
        f"{datetime.now(UTC).year}"
    )


if __name__ == "__main__":
    main()
