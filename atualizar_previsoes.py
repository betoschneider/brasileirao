"""Atualiza as previsões do Brasileirão Série A.

Script que substitui o antigo notebook ``brasileirao-analise.ipynb``. Ele:

1. coleta as partidas da edição atual (API do IG) e das temporadas anteriores;
2. enriquece cada partida com aproveitamento, médias de gols, histórico de
   confrontos e valor de mercado dos times;
3. valida e treina os modelos de classificação (LOSO e/ou Monte Carlo), projeta
   as rodadas restantes e calcula a probabilidade de cada cenário final;
4. grava ``app/tabela-evolucao-modelos.csv``, ``app/partidas-modelos.csv``,
   ``app/times-stats.csv`` e ``app/chances-times.csv``, consumidos pelo
   dashboard Streamlit.

Uso:
    python atualizar_previsoes.py [--n-splits 200] [--n-jobs -1] [--loso | --sem-loso] [--mccv | --sem-mccv] [--verbose]
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
    from joblib import Parallel, delayed
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
except ImportError:
    requests = None
    Parallel = None
    delayed = None
    GradientBoostingClassifier = None
    RandomForestClassifier = None
    VotingClassifier = None
    LogisticRegression = None
    accuracy_score = None
    train_test_split = None
    KNeighborsClassifier = None

logger = logging.getLogger("atualizar_previsoes")

################################
# Configurações do pipeline    #
################################

# Escolha dos modelos/validações a executar:
# Alterne para True ou False para habilitar/desabilitar cada validação.
EXECUTAR_LOSO = False  # Deixe False para desativar a validação Leave-One-Season-Out (LOSO)
EXECUTAR_MCCV = True  # Deixe False para desativar a validação Monte Carlo (MCCV)

RAIZ = Path(__file__).resolve().parent
DIR_APP = RAIZ / "app"

ARQUIVO_VALOR_TIME = DIR_APP / "valor-time-titular.csv"
ARQUIVO_TABELA_EVOLUCAO = DIR_APP / "tabela-evolucao-modelos.csv"
ARQUIVO_PARTIDAS = DIR_APP / "partidas-modelos.csv"
ARQUIVO_TIMES_STATS = DIR_APP / "times-stats.csv"
ARQUIVO_CHANCES = DIR_APP / "chances-times.csv"

URL_FASE_ATUAL = "https://service.ig.com.br/football_ig/campeonatos/10/fases/921"
URL_API_HISTORICO = "https://api-brasileirao.betoschneider.com"

CABECALHOS = {"User-Agent": "brasileirao-previsoes/1.0"}
TEMPO_LIMITE_REQUISICAO = 30  # segundos

JANELA_CONFRONTO = 5  # confrontos diretos considerados no histórico
JANELA_RECENTE = 5  # partidas consideradas no aproveitamento recente
# 200 splits já dão ~0,5% de resolução na confiança de cada partida e uma
# estimativa estável da acurácia, rodando em ~10 min. Use --n-splits para mudar.
N_SPLITS_MCCV = 200
N_JOBS = -1  # núcleos usados na validação Monte Carlo (-1 = todos)
RANDOM_STATE = 42

COLUNAS_DESCARTAR_PARTIDAS = [
    "disputa_penalti",
    "slug",
    "hora_realizacao",
    "data_realizacao_iso",
    "_link",
    "edicao.nome",
    "edicao.slug",
    "estadio",
]

MAPEAMENTO_HISTORICO = {
    "temporada": "edicao.temporada",
    "time_mandante_nome_popular": "time_mandante.nome_popular",
    "time_visitante_nome_popular": "time_visitante.nome_popular",
    "estadio_nome_popular": "estadio.nome_popular",
}

COLUNAS_MANDANTE = [
    "rodada",
    "rodada_num",
    "data_realizacao",
    "partida_id",
    "estadio.estadio_id",
    "estadio.nome_popular",
    "status",
    "edicao.edicao_id",
    "edicao.temporada",
    "edicao.nome_popular",
    "campeonato.campeonato_id",
    "campeonato.nome",
    "campeonato.slug",
    "time_mandante.time_id",
    "time_mandante.nome_popular",
    "time_visitante.time_id",
    "time_visitante.nome_popular",
    "placar_mandante",
    "placar_visitante",
    "time_mandante.valor",
    "time_visitante.valor",
]

COLUNAS_VISITANTE = [
    "rodada",
    "rodada_num",
    "data_realizacao",
    "partida_id",
    "estadio.estadio_id",
    "estadio.nome_popular",
    "status",
    "edicao.edicao_id",
    "edicao.temporada",
    "edicao.nome_popular",
    "campeonato.campeonato_id",
    "campeonato.nome",
    "campeonato.slug",
    "time_mandante.time_id",
    "time_mandante.nome_popular",
    "time_visitante.time_id",
    "time_visitante.nome_popular",
    "placar_visitante",
    "placar_mandante",
    "time_mandante.valor",
    "time_visitante.valor",
]

RENOMEAR_MANDANTE = {
    "time_mandante.time_id": "time_id",
    "time_mandante.nome_popular": "time",
    "time_visitante.time_id": "adversario_id",
    "time_visitante.nome_popular": "adversario",
    "time_mandante.valor": "valor_time",
    "time_visitante.valor": "valor_adversario",
    "placar_mandante": "gols_time",
    "placar_visitante": "gols_adversario",
}

RENOMEAR_VISITANTE = {
    "time_mandante.time_id": "adversario_id",
    "time_mandante.nome_popular": "adversario",
    "time_visitante.time_id": "time_id",
    "time_visitante.nome_popular": "time",
    "time_mandante.valor": "valor_adversario",
    "time_visitante.valor": "valor_time",
    "placar_visitante": "gols_time",
    "placar_mandante": "gols_adversario",
}

FEATURES = [
    "mandante",
    "rodada_num",
    "aproveitamento",
    "aproveitamento_recente",
    "aproveitamento_adversario",
    "aproveitamento_recente_adversario",
    "media_gols_marcados",
    "media_gols_sofridos",
    "media_gols_marcados_adversario",
    "media_gols_sofridos_adversario",
    "diff_media_gols",
    "diff_media_gols_sofridos",
    "std_gols_marcados",
    "std_gols_sofridos",
    "std_gols_marcados_adversario",
    "std_gols_sofridos_adversario",
]


#########################
# Coleta de dados (API) #
#########################


def requisitar_json(url: str, **params) -> object:
    """Faz uma requisição GET e devolve o JSON, validando o certificado TLS."""
    resposta = requests.get(
        url,
        headers=CABECALHOS,
        params=params or None,
        timeout=TEMPO_LIMITE_REQUISICAO,
    )
    resposta.raise_for_status()
    return resposta.json()


def coletar_partidas_atuais() -> pd.DataFrame:
    """Coleta as partidas e os dados da edição atual do Brasileirão."""
    dados = requisitar_json(URL_FASE_ATUAL)
    edicao = {
        f"edicao.{chave}": valor for chave, valor in dados.get("edicao", {}).items()
    }

    registros = []
    for rodada, partidas_da_rodada in dados.get("partidas", {}).items():
        for partida in partidas_da_rodada:
            registros.append({"rodada": rodada, **partida, **edicao})

    partidas_df = pd.json_normalize(registros)
    partidas_df = partidas_df.drop(
        columns=[c for c in COLUNAS_DESCARTAR_PARTIDAS if c in partidas_df.columns]
    )
    partidas_df["data_realizacao"] = pd.to_datetime(
        partidas_df["data_realizacao"], format="%d/%m/%Y", errors="coerce"
    )
    return partidas_df


def coletar_historico(ano_fim: int) -> pd.DataFrame:
    """Coleta as partidas finalizadas de todas as temporadas até ``ano_fim``."""
    ano_inicio = requisitar_json(f"{URL_API_HISTORICO}/temporadas/min")["ano"]

    registros = []
    for ano in range(ano_inicio, ano_fim + 1):
        dados = requisitar_json(f"{URL_API_HISTORICO}/partidas/ano/{ano}")
        if isinstance(dados, list):
            registros.extend(dados)
        else:
            logger.warning("Temporada %s retornou um formato inesperado.", ano)

    historico_df = pd.DataFrame(registros).rename(columns=MAPEAMENTO_HISTORICO)
    historico_df["data_realizacao"] = pd.to_datetime(
        historico_df["data_realizacao"], errors="coerce"
    )
    return historico_df


#####################
# ETL e features    #
#####################


def consolidar_partidas(
    partidas_df: pd.DataFrame, historico_df: pd.DataFrame
) -> pd.DataFrame:
    """Junta o histórico ao campeonato atual e adiciona temporada/rodada e valores."""
    for coluna in partidas_df.columns.difference(historico_df.columns):
        historico_df[coluna] = np.nan
    historico_df = historico_df[partidas_df.columns]

    partidas_df = pd.concat([historico_df, partidas_df], ignore_index=True)

    partidas_df["edicao.temporada"] = pd.to_numeric(
        partidas_df["edicao.temporada"], errors="coerce"
    ).astype("int64")
    partidas_df["rodada_num"] = (
        partidas_df["rodada"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )

    valor_time_df = pd.read_csv(ARQUIVO_VALOR_TIME)
    for momento, coluna_origem in (
        ("mandante", "time_mandante.nome_popular"),
        ("visitante", "time_visitante.nome_popular"),
    ):
        partidas_df = partidas_df.merge(
            valor_time_df,
            left_on=[coluna_origem, "edicao.temporada"],
            right_on=["time", "ano"],
            how="left",
        )
        partidas_df = partidas_df.drop(columns=["time", "ano"]).rename(
            columns={"time.valor": f"time_{momento}.valor"}
        )

    return partidas_df


def separar_visoes(partidas_df: pd.DataFrame) -> pd.DataFrame:
    """Cria duas linhas por partida: uma na ótica do mandante e outra do visitante."""
    mandante = partidas_df[COLUNAS_MANDANTE].copy()
    mandante["mandante"] = 1
    mandante = mandante.rename(columns=RENOMEAR_MANDANTE)

    visitante = partidas_df[COLUNAS_VISITANTE].copy()
    visitante["mandante"] = 0
    visitante = visitante.rename(columns=RENOMEAR_VISITANTE)

    partidas_model_df = pd.concat([mandante, visitante], ignore_index=True)
    return partidas_model_df.sort_values(
        by=["data_realizacao", "rodada_num", "partida_id", "mandante"],
        ascending=[True, True, True, False],
    )


def adicionar_medias_gols(partidas_model_df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona média e desvio padrão de gols marcados/sofridos até a rodada anterior."""
    estatisticas = None
    for ano_rodada in partidas_model_df["ano-rodada"].drop_duplicates():
        temporada, rodada = (int(valor) for valor in ano_rodada.split("."))
        anteriores = partidas_model_df[
            (partidas_model_df["rodada_num"] < rodada)
            & (partidas_model_df["edicao.temporada"] == temporada)
        ]
        if anteriores.empty:
            continue

        temp = (
            anteriores[["time", "gols_time", "gols_adversario"]]
            .groupby("time")
            .agg(["mean", "std"])
            .reset_index()
        )
        temp.columns = [
            "time",
            "media_gols_marcados",
            "std_gols_marcados",
            "media_gols_sofridos",
            "std_gols_sofridos",
        ]
        temp["ano-rodada"] = ano_rodada
        estatisticas = pd.concat([estatisticas, temp], ignore_index=True)

    if estatisticas is None:
        raise RuntimeError(
            "Não há partidas finalizadas suficientes para calcular as médias de gols."
        )

    # std é NaN quando o time tem apenas uma partida; assume 0 nesses casos.
    estatisticas = estatisticas.fillna(0)

    partidas_model_df = partidas_model_df.merge(
        estatisticas, on=["ano-rodada", "time"], how="left"
    )
    estatisticas = estatisticas.rename(
        columns={
            "time": "adversario",
            "media_gols_marcados": "media_gols_marcados_adversario",
            "std_gols_marcados": "std_gols_marcados_adversario",
            "media_gols_sofridos": "media_gols_sofridos_adversario",
            "std_gols_sofridos": "std_gols_sofridos_adversario",
        }
    )
    return partidas_model_df.merge(
        estatisticas, on=["ano-rodada", "adversario"], how="left"
    )


def adicionar_pontos(partidas_model_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula os pontos conquistados por cada time até a rodada anterior."""
    partidas_model_df["pontos"] = partidas_model_df.apply(
        lambda x: (
            3
            if x["gols_time"] > x["gols_adversario"]
            else (
                1
                if x["gols_time"] == x["gols_adversario"]
                else (0 if x["status"] != "agendado" else np.nan)
            )
        ),
        axis=1,
    )

    pontos_df = None
    for ano_rodada in partidas_model_df["ano-rodada"].drop_duplicates():
        temporada, rodada = (int(valor) for valor in ano_rodada.split("."))
        temp = partidas_model_df[
            (partidas_model_df["rodada_num"] < rodada)
            & (partidas_model_df["edicao.temporada"] == temporada)
        ]
        temp = temp[["time", "pontos"]].groupby("time").sum().reset_index()
        temp["ano-rodada"] = ano_rodada
        pontos_df = pd.concat([pontos_df, temp], ignore_index=True)

    pontos_df = pontos_df.rename(columns={"pontos": "pontos_time"})
    partidas_model_df = partidas_model_df.merge(
        pontos_df, on=["ano-rodada", "time"], how="left"
    )
    pontos_df = pontos_df.rename(
        columns={"pontos_time": "pontos_adversario", "time": "adversario"}
    )
    return partidas_model_df.merge(
        pontos_df, on=["ano-rodada", "adversario"], how="left"
    )


def adicionar_aproveitamento(partidas_model_df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona aproveitamento acumulado e o aproveitamento das últimas partidas, do time e do adversário."""
    partidas_model_df["jogos"] = np.nan
    realizadas = partidas_model_df["status"] != "agendado"
    partidas_model_df.loc[realizadas, "jogos"] = (
        partidas_model_df.loc[realizadas]
        .groupby(["edicao.temporada", "time"])
        .cumcount()
    )
    partidas_model_df["jogos"] = (
        partidas_model_df.groupby(["edicao.temporada", "time"])["jogos"]
        .ffill()
        .fillna(0)
        .astype(int)
    )

    partidas_model_df["pontos_acumulados"] = partidas_model_df.groupby(
        ["edicao.temporada", "time"]
    )["pontos_time"].cumsum()
    partidas_model_df["aproveitamento"] = partidas_model_df["pontos_time"] / (
        (partidas_model_df["jogos"] + 1) * 3
    )

    partidas_model_df = partidas_model_df.sort_values(
        ["edicao.temporada", "time", "rodada_num"]
    )
    pontos_recentes = []
    jogos_validos = []
    for _, grupo in partidas_model_df.groupby(["edicao.temporada", "time"]):
        ultimos = deque()
        for _, linha in grupo.iterrows():
            # considera apenas as partidas anteriores à rodada atual
            pontos_recentes.append(sum(ultimos))
            jogos_validos.append(len(ultimos))
            if linha["status"] != "agendado":
                ultimos.append(linha["pontos"])
            while len(ultimos) > JANELA_RECENTE:
                ultimos.popleft()

    partidas_model_df["pontos_recentes"] = pontos_recentes
    partidas_model_df["jogos_validos"] = jogos_validos
    partidas_model_df["aproveitamento_recente"] = partidas_model_df[
        "pontos_recentes"
    ] / (partidas_model_df["jogos_validos"] * 3)

    adversario = partidas_model_df[
        [
            "edicao.temporada",
            "rodada_num",
            "time",
            "aproveitamento",
            "aproveitamento_recente",
        ]
    ].rename(
        columns={
            "time": "adversario",
            "aproveitamento": "aproveitamento_adversario",
            "aproveitamento_recente": "aproveitamento_recente_adversario",
        }
    )
    return partidas_model_df.merge(
        adversario, on=["edicao.temporada", "rodada_num", "adversario"], how="left"
    )


def adicionar_confronto_recente(
    partidas_model_df: pd.DataFrame, janela: int = JANELA_CONFRONTO
) -> pd.DataFrame:
    """Adiciona o aproveitamento do time nos últimos confrontos diretos contra o adversário."""
    df = partidas_model_df.reset_index(drop=True).copy()
    confrontos = (
        df[["edicao.temporada", "rodada_num", "time", "adversario", "pontos", "status"]]
        .sort_values(["time", "adversario", "edicao.temporada", "rodada_num"])
        .reset_index(drop=True)
    )

    valores = {}
    for _, grupo in confrontos.groupby(["time", "adversario"], sort=False):
        ultimos = deque()
        for indice, linha in grupo.iterrows():
            valores[indice] = sum(ultimos) / len(ultimos) if ultimos else np.nan
            if linha["status"] != "agendado":
                ultimos.append(int(linha["pontos"]))
            if len(ultimos) > janela:
                ultimos.popleft()

    confrontos["confronto_recente"] = confrontos.index.map(valores)
    out = df.merge(
        confrontos[
            [
                "edicao.temporada",
                "rodada_num",
                "time",
                "adversario",
                "confronto_recente",
            ]
        ],
        on=["edicao.temporada", "rodada_num", "time", "adversario"],
        how="left",
    )

    out_adv = out[
        ["edicao.temporada", "rodada_num", "time", "adversario", "confronto_recente"]
    ].rename(
        columns={
            "time": "adversario",
            "adversario": "time",
            "confronto_recente": "confronto_recente_adversario",
        }
    )
    out = out.merge(
        out_adv, on=["edicao.temporada", "rodada_num", "time", "adversario"], how="left"
    )
    out["confronto_recente"] = out["confronto_recente"].fillna(0)
    out["confronto_recente_adversario"] = out["confronto_recente_adversario"].fillna(0)
    return out


def enriquecer_agendadas(
    partidas_model_df: pd.DataFrame, partidas_agendadas_df: pd.DataFrame
) -> pd.DataFrame:
    """Copia as estatísticas mais recentes de cada time para as partidas agendadas."""
    df_max_rodada = (
        partidas_model_df.groupby(["edicao.temporada", "time"], as_index=False)[
            "rodada_num"
        ].max()
    ).merge(
        partidas_model_df, on=["edicao.temporada", "time", "rodada_num"], how="left"
    )

    colunas = [
        "edicao.temporada",
        "time",
        "media_gols_marcados",
        "std_gols_marcados",
        "media_gols_sofridos",
        "std_gols_sofridos",
        "pontos",
        "pontos_time",
        "jogos",
        "pontos_acumulados",
        "aproveitamento",
        "pontos_recentes",
        "jogos_validos",
        "aproveitamento_recente",
        "confronto_recente",
    ]
    df_max_rodada = df_max_rodada[colunas]
    partidas_agendadas_df = partidas_agendadas_df.merge(
        df_max_rodada, on=["edicao.temporada", "time"], how="left"
    )

    df_max_rodada = df_max_rodada.rename(
        columns={
            "time": "adversario",
            "media_gols_marcados": "media_gols_marcados_adversario",
            "media_gols_sofridos": "media_gols_sofridos_adversario",
            "pontos_time": "pontos_adversario",
            "aproveitamento": "aproveitamento_adversario",
            "aproveitamento_recente": "aproveitamento_recente_adversario",
            "confronto_recente": "confronto_recente_adversario",
        }
    ).drop(
        columns=[
            "pontos",
            "jogos",
            "pontos_acumulados",
            "pontos_recentes",
            "jogos_validos",
        ]
    )

    return partidas_agendadas_df.merge(
        df_max_rodada, on=["edicao.temporada", "adversario"], how="left"
    )


COLUNAS_DESCARTAR_MODELO = [
    "rodada",
    "estadio.nome_popular",
    "edicao.edicao_id",
    "ano-rodada",
    "edicao.nome_popular",
    "campeonato.campeonato_id",
    "campeonato.nome",
    "campeonato.slug",
    "jogos",
    "pontos_acumulados",
    "pontos_validos",
    "pontos_recentes",
    "jogos_validos",
    "pontos_shifted",
]


def preparar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa o ETL completo e devolve os dados prontos para a modelagem.

    Retorna ``(partidas_model_final_df, partidas_finalizadas_df)``, onde o
    primeiro contém todas as partidas (encerradas de todas as temporadas e
    agendadas da temporada atual) e o segundo apenas os jogos já encerrados da
    temporada atual, usados como ponto de partida da tabela de evolução.
    """
    partidas_df = coletar_partidas_atuais()
    temporada_atual = int(partidas_df["edicao.temporada"].iloc[0])
    logger.info("Temporada atual: %s", temporada_atual)

    historico_df = coletar_historico(ano_fim=temporada_atual - 1)
    logger.info("Histórico coletado: %s partidas", len(historico_df))

    partidas_df = consolidar_partidas(partidas_df, historico_df)
    partidas_model_df = separar_visoes(partidas_df)

    partidas_agendadas_df = partidas_model_df[
        partidas_model_df["status"] != "finalizado"
    ]
    partidas_model_df = partidas_model_df[partidas_model_df["status"] == "finalizado"]
    partidas_model_df["ano-rodada"] = (
        partidas_model_df["edicao.temporada"].astype(str)
        + "."
        + partidas_model_df["rodada_num"].astype(str)
    )

    partidas_model_df = adicionar_medias_gols(partidas_model_df)
    partidas_model_df = adicionar_pontos(partidas_model_df)
    partidas_model_df = adicionar_aproveitamento(partidas_model_df)
    partidas_model_df = adicionar_confronto_recente(partidas_model_df)
    partidas_agendadas_df = enriquecer_agendadas(
        partidas_model_df, partidas_agendadas_df
    )

    partidas_model_df = pd.concat(
        [partidas_model_df, partidas_agendadas_df], ignore_index=True
    ).drop(
        columns=[c for c in COLUNAS_DESCARTAR_MODELO if c in partidas_model_df.columns]
    )

    partidas_finalizadas_df = partidas_model_df[
        (partidas_model_df["status"] != "agendado")
        & (
            partidas_model_df["edicao.temporada"]
            == partidas_model_df["edicao.temporada"].max()
        )
    ]
    # a rodada 1 não tem histórico anterior, então fica fora do treino
    partidas_model_df = partidas_model_df[
        partidas_model_df["rodada_num"] != 1
    ].reset_index(drop=True)

    logger.info(
        "Base de modelagem: %s partidas (%s encerradas na temporada atual)",
        len(partidas_model_df),
        len(partidas_finalizadas_df),
    )
    return partidas_model_df, partidas_finalizadas_df


########################
# Treino e projeção    #
########################


def criar_classificadores() -> dict:
    """Cria os modelos com sementes fixas, garantindo resultados reprodutíveis.

    Os modelos usam um único núcleo porque a validação Monte Carlo já é
    paralelizada em nível de split (evita sobrecarregar a máquina).
    """
    return {
        "VotingClassifier": VotingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
                (
                    "gb",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ],
            voting="soft",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona as features e cria as diferenças entre time e adversário."""
    df = df.copy()
    df["diff_media_gols"] = (
        df["media_gols_marcados"] - df["media_gols_marcados_adversario"]
    )
    df["diff_media_gols_sofridos"] = (
        df["media_gols_sofridos"] - df["media_gols_sofridos_adversario"]
    )

    X = df[FEATURES].fillna(0)
    X["mandante"] = X["mandante"].astype(int)
    return X


def validar_loso(
    X_realizados: pd.DataFrame,
    y_realizados: pd.Series,
    temporada_realizados: pd.Series,
    temporadas: list,
) -> dict:
    """Valida cada modelo deixando uma temporada de fora por vez.

    Em cada rodada de validação, treina com as temporadas anteriores e testa
    com a temporada seguinte, garantindo que o modelo nunca veja o futuro.
    """
    resultados = {}
    for nome, clf in criar_classificadores().items():
        acuracias = []
        for i in range(1, len(temporadas)):
            treino = temporada_realizados.isin(temporadas[:i])
            teste = temporada_realizados == temporadas[i]
            clf.fit(X_realizados[treino], y_realizados[treino])
            acuracias.append(clf.score(X_realizados[teste], y_realizados[teste]))
        resultados[nome] = (float(np.mean(acuracias)), float(np.std(acuracias)))
        logger.info("LOSO %-18s acc=%.3f ± %.3f", nome, *resultados[nome])
    return resultados


def executar_split(
    indice: int,
    X_realizados: pd.DataFrame,
    y_realizados: pd.Series,
    X_agendadas: pd.DataFrame,
) -> dict:
    """Treina todos os modelos em um split da validação Monte Carlo.

    Devolve ``{modelo: (acurácia, previsões das partidas agendadas)}``.
    """
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_realizados,
        y_realizados,
        test_size=0.2,
        stratify=y_realizados,
        random_state=indice,
    )
    resultado = {}
    for nome, clf in criar_classificadores().items():
        clf.fit(X_treino, y_treino)
        resultado[nome] = (
            accuracy_score(y_teste, clf.predict(X_teste)),
            clf.predict(X_agendadas),
        )
    return resultado


def calcular_chances(
    simulacoes: np.ndarray,
    times_agendadas: np.ndarray,
    pontos_atuais: pd.Series,
    pontos_projetados: pd.Series,
    modelo: str,
) -> pd.DataFrame:
    """Transforma as simulações Monte Carlo em probabilidades por time.

    ``simulacoes`` é uma matriz ``partidas agendadas x splits`` com os pontos
    (0, 1 ou 3) previstos para cada partida em cada simulação. Somando esses
    pontos ao que cada time já conquistou, chega-se à distribuição da
    pontuação final, de onde saem a chance da pontuação prevista acontecer, a
    faixa provável e as probabilidades de título, G4 e rebaixamento.
    """
    times = list(pontos_atuais.index)
    posicao_do_time = {nome_time: indice for indice, nome_time in enumerate(times)}
    n_splits = simulacoes.shape[1]

    ganhos = np.zeros((len(times), n_splits))
    vitorias = np.zeros((len(times), n_splits))
    for nome_time in times:
        partidas_do_time = simulacoes[times_agendadas == nome_time]
        if partidas_do_time.size:
            indice = posicao_do_time[nome_time]
            # soma as partidas do time (eixo 0) mantendo um valor por split
            ganhos[indice] = partidas_do_time.sum(axis=0)
            vitorias[indice] = (partidas_do_time == 3).sum(axis=0)

    pontos = ganhos + pontos_atuais.to_numpy()[:, None]

    # posição em cada simulação, com desempate por vitórias (como na tabela real)
    posicoes = np.empty((len(times), n_splits), dtype=int)
    for split in range(n_splits):
        ordem = np.lexsort((vitorias[:, split], pontos[:, split]))[::-1]
        posicoes[ordem, split] = np.arange(1, len(times) + 1)

    # limites algébricos do campeonato (garantia ou eliminação matemática)
    jogos_restantes = {
        nome_time: int(np.sum(times_agendadas == nome_time)) for nome_time in times
    }
    pontos_max = {
        nome_time: float(pontos_atuais[nome_time] + 3 * jogos_restantes[nome_time])
        for nome_time in times
    }
    pontos_min = {nome_time: float(pontos_atuais[nome_time]) for nome_time in times}

    status_mat = {}
    for t in times:
        outros = [outro for outro in times if outro != t]
        outros_max_desc = sorted([pontos_max[outro] for outro in outros], reverse=True)
        outros_min_desc = sorted([pontos_min[outro] for outro in outros], reverse=True)

        # Título
        if pontos_min[t] > outros_max_desc[0]:
            st_titulo = "garantido"
        elif pontos_max[t] < outros_min_desc[0]:
            st_titulo = "eliminado"
        else:
            st_titulo = "aberto"

        # G4
        if pontos_min[t] > outros_max_desc[3]:
            st_g4 = "garantido"
        elif pontos_max[t] < outros_min_desc[3]:
            st_g4 = "eliminado"
        else:
            st_g4 = "aberto"

        # Rebaixamento (Z4)
        if pontos_max[t] < outros_min_desc[15]:
            st_z4 = "rebaixado"
        elif pontos_min[t] > outros_max_desc[15]:
            st_z4 = "salvo"
        else:
            st_z4 = "aberto"

        status_mat[t] = (st_titulo, st_g4, st_z4)

    linhas = []
    for nome_time in times:
        indice = posicao_do_time[nome_time]
        valores = pontos[indice]
        previsto = int(pontos_projetados[nome_time])
        st_titulo, st_g4, st_z4 = status_mat[nome_time]
        linhas.append(
            {
                "modelo": modelo,
                "time": nome_time,
                "pontos_previstos": previsto,
                "chance_pontos_previstos": float(np.mean(valores == previsto)),
                "pontos_p10": int(np.percentile(valores, 10)),
                "pontos_p90": int(np.percentile(valores, 90)),
                "chance_titulo": float(np.mean(posicoes[indice] == 1)),
                "chance_g4": float(np.mean(posicoes[indice] <= 4)),
                "chance_rebaixamento": float(
                    np.mean(posicoes[indice] >= len(times) - 3)
                ),
                "status_titulo": st_titulo,
                "status_g4": st_g4,
                "status_rebaixamento": st_z4,
            }
        )
    return pd.DataFrame(linhas)


def treinar_e_prever(
    partidas_model_final_df: pd.DataFrame,
    partidas_finalizadas_df: pd.DataFrame,
    n_splits: int = N_SPLITS_MCCV,
    n_jobs: int = N_JOBS,
    executar_loso: bool = EXECUTAR_LOSO,
    executar_mccv: bool = EXECUTAR_MCCV,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Treina os modelos, projeta as rodadas restantes e monta as saídas do dashboard."""
    if not executar_loso and not executar_mccv:
        raise ValueError(
            "Ao menos uma validação (LOSO ou MCCV) deve estar habilitada."
        )

    df_realizados = (
        partidas_model_final_df[partidas_model_final_df["status"] != "agendado"]
        .dropna(subset=["pontos"])
        .copy()
    )
    df_realizados["pontos"] = df_realizados["pontos"].astype(int)
    df_agendadas = partidas_model_final_df[
        partidas_model_final_df["status"] == "agendado"
    ].copy()

    # pontuação já conquistada por cada time na temporada atual
    pontos_atuais = partidas_finalizadas_df.groupby("time")["pontos"].sum()

    X_realizados = preparar_features(df_realizados)
    y_realizados = df_realizados["pontos"]
    X_agendadas = preparar_features(df_agendadas)

    # mantém a ordem das partidas agendadas igual à ordem das previsões
    # (X_agendadas), para que a atribuição abaixo seja sempre correta
    resumo_agendadas = df_agendadas[
        ["rodada_num", "partida_id", "mandante", "time", "adversario"]
    ].reset_index(drop=True)

    ####################################
    # Validação Leave-One-Season-Out   #
    ####################################
    melhor_loso = None
    loso_resumo_agendadas_df = None
    if executar_loso:
        logger.info("Rodando validação Leave-One-Season-Out (LOSO)...")
        temporadas = sorted(df_realizados["edicao.temporada"].unique())
        resultados_loso = validar_loso(
            X_realizados, y_realizados, df_realizados["edicao.temporada"], temporadas
        )
        melhor_loso = max(resultados_loso, key=lambda nome: resultados_loso[nome][0])
        loso_resumo_agendadas_df = resumo_agendadas.copy()

        classificador = criar_classificadores()[melhor_loso]
        classificador.fit(X_realizados, y_realizados)
        loso_resumo_agendadas_df[f"{melhor_loso}_pred"] = classificador.predict(X_agendadas)
        logger.info("Melhor modelo (LOSO): %s", melhor_loso)

    ####################################
    # Validação Monte Carlo (MCCV)     #
    ####################################
    melhor_mccv = None
    mccv_resumo_agendadas_df = None
    chances_df = pd.DataFrame()
    pontos_projetados = None

    if executar_mccv:
        classificadores = criar_classificadores()

        logger.info("Rodando validação Monte Carlo (%s splits)...", n_splits)
        # cada split é independente e reprodutível (random_state=i), então pode
        # rodar em paralelo sem alterar o resultado final
        resultados_splits = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(executar_split)(i, X_realizados, y_realizados, X_agendadas)
            for i in range(n_splits)
        )

        acuracias = {nome: [] for nome in classificadores}
        predicoes = {nome: [] for nome in classificadores}
        for resultado in resultados_splits:
            for nome, (acuracia, predicao) in resultado.items():
                acuracias[nome].append(acuracia)
                predicoes[nome].append(predicao)

        medias = {nome: float(np.mean(valores)) for nome, valores in acuracias.items()}
        for nome, media in medias.items():
            logger.info(
                "MCCV %-18s acc=%.3f ± %.3f", nome, media, float(np.std(acuracias[nome]))
            )
        melhor_mccv = max(medias, key=medias.get)
        logger.info("Melhor modelo (MCCV): %s", melhor_mccv)

        mccv_resumo_agendadas_df = resumo_agendadas.copy()
        todas_predicoes = np.array(predicoes[melhor_mccv]).T  # partidas x splits
        mccv_resumo_agendadas_df[f"{melhor_mccv}_pred"] = [
            int(np.bincount(linha.astype(int)).argmax()) for linha in todas_predicoes
        ]
        mccv_resumo_agendadas_df[f"{melhor_mccv}_freq"] = [
            {ponto: (np.sum(linha == ponto) / n_splits) * 100 for ponto in (0, 1, 3)}
            for linha in todas_predicoes
        ]
        mccv_resumo_agendadas_df[f"{melhor_mccv}_pred_freq"] = (
            mccv_resumo_agendadas_df.apply(
                lambda linha: linha[f"{melhor_mccv}_freq"].get(
                    linha[f"{melhor_mccv}_pred"], 0
                ),
                axis=1,
            )
        )

        ####################################
        # Chances por time                 #
        ####################################
        # pontuação prevista: soma, por time, do resultado mais provável de cada
        # partida restante (é o mesmo número exibido na tabela de evolução)
        ganhos_previstos = mccv_resumo_agendadas_df.groupby("time")[
            f"{melhor_mccv}_pred"
        ].sum()
        pontos_projetados = (
            pontos_atuais + ganhos_previstos.reindex(pontos_atuais.index).fillna(0)
        ).astype(int)

        chances_df = calcular_chances(
            simulacoes=todas_predicoes,
            times_agendadas=df_agendadas["time"].to_numpy(),
            pontos_atuais=pontos_atuais,
            pontos_projetados=pontos_projetados,
            modelo=f"{melhor_mccv}_MCCV",
        )

    ####################################
    # Tabela de evolução por rodada    #
    ####################################
    modelos_evolucao = []
    if executar_loso and melhor_loso is not None and loso_resumo_agendadas_df is not None:
        modelos_evolucao.append((melhor_loso, "LOSO", loso_resumo_agendadas_df))
    if executar_mccv and melhor_mccv is not None and mccv_resumo_agendadas_df is not None:
        modelos_evolucao.append((melhor_mccv, "MCCV", mccv_resumo_agendadas_df))

    tabela_evolucao_df = None
    for modelo, validacao, origem in modelos_evolucao:
        coluna_pred = f"{modelo}_pred"
        temp_df = (
            pd.concat(
                [
                    partidas_finalizadas_df[["rodada_num", "time", "pontos"]],
                    origem[["rodada_num", "time", coluna_pred]].rename(
                        columns={coluna_pred: "pontos"}
                    ),
                ],
                ignore_index=True,
            )
            .sort_values(by=["time", "rodada_num"])
            .reset_index(drop=True)
        )

        temp_df["vitoria"] = (temp_df["pontos"] == 3).astype(int)
        temp_df["pontos"] = temp_df.groupby("time")["pontos"].cumsum()
        temp_df["vitorias"] = temp_df.groupby("time")["vitoria"].cumsum()
        temp_df["modelo"] = f"{modelo}_{validacao}"
        tabela_evolucao_df = pd.concat([tabela_evolucao_df, temp_df], ignore_index=True)

    tabela_evolucao_df = tabela_evolucao_df.sort_values(
        by=["rodada_num", "modelo", "pontos", "vitorias"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    tabela_evolucao_df["posicao"] = (
        tabela_evolucao_df.groupby(["rodada_num", "modelo"]).cumcount() + 1
    )
    tabela_evolucao_df["data_atualizacao"] = pd.Timestamp.now()

    # conferência: a pontuação final da tabela deve ser a mesma dos pontos
    # projetados a partir das previsões partida a partida (quando MCCV estiver ativo)
    if executar_mccv and melhor_mccv is not None and pontos_projetados is not None:
        rodada_final = int(tabela_evolucao_df["rodada_num"].max())
        pontuacao_final = (
            tabela_evolucao_df[
                (tabela_evolucao_df["rodada_num"] == rodada_final)
                & (tabela_evolucao_df["modelo"] == f"{melhor_mccv}_MCCV")
            ]
            .set_index("time")["pontos"]
            .astype(int)
        )
        if not pontuacao_final.sort_index().equals(pontos_projetados.sort_index()):
            logger.warning("A pontuação projetada divergiu da tabela de evolução.")

    ####################################
    # Tabela de partidas               #
    ####################################
    partidas_finalizadas_df = partidas_finalizadas_df[
        ["rodada_num", "mandante", "time", "adversario", "pontos"]
    ].copy()
    partidas_finalizadas_df["status"] = "Finalizado"

    lista_predicoes = []
    if (
        executar_loso
        and melhor_loso is not None
        and loso_resumo_agendadas_df is not None
    ):
        loso_df = loso_resumo_agendadas_df[
            ["rodada_num", "mandante", "time", "adversario", f"{melhor_loso}_pred"]
        ].rename(columns={f"{melhor_loso}_pred": "pontos"})
        loso_df["validacao"] = "LOSO"
        loso_df["modelo"] = f"{melhor_loso}_LOSO"
        lista_predicoes.append(loso_df)

    if (
        executar_mccv
        and melhor_mccv is not None
        and mccv_resumo_agendadas_df is not None
    ):
        mccv_df = mccv_resumo_agendadas_df[
            [
                "rodada_num",
                "mandante",
                "time",
                "adversario",
                f"{melhor_mccv}_pred",
                f"{melhor_mccv}_pred_freq",
            ]
        ].rename(
            columns={
                f"{melhor_mccv}_pred": "pontos",
                f"{melhor_mccv}_pred_freq": "frequencia",
            }
        )
        mccv_df["validacao"] = "MCCV"
        mccv_df["modelo"] = f"{melhor_mccv}_MCCV"
        lista_predicoes.append(mccv_df)

    if lista_predicoes:
        partidas_predicao_df = pd.concat(lista_predicoes, ignore_index=True)
        partidas_predicao_df["status"] = "Previsto"
        partidas_df = pd.concat([partidas_predicao_df, partidas_finalizadas_df], ignore_index=True)
    else:
        partidas_df = partidas_finalizadas_df.copy()

    partidas_df["resultado"] = partidas_df["pontos"].apply(
        lambda pontos: (
            "Vitória" if pontos == 3 else ("Empate" if pontos == 1 else "Derrota")
        )
    )
    partidas_df["mandante"] = partidas_df["mandante"].apply(
        lambda valor: "Sim" if valor == 1 else "Não"
    )
    if "frequencia" not in partidas_df.columns:
        partidas_df["frequencia"] = np.nan
    if "modelo" not in partidas_df.columns:
        partidas_df["modelo"] = np.nan

    partidas_df = partidas_df.sort_values(
        by=["rodada_num", "time", "mandante"], ascending=[True, True, False]
    )[
        [
            "rodada_num",
            "status",
            "mandante",
            "time",
            "adversario",
            "resultado",
            "frequencia",
            "modelo",
        ]
    ]

    ####################################
    # Estatísticas dos times           #
    ####################################
    colunas_stats = [
        "time",
        "pontos_time",
        "aproveitamento",
        "aproveitamento_recente",
        "media_gols_marcados",
        "media_gols_sofridos",
    ]
    times_stats = partidas_model_final_df[
        partidas_model_final_df["status"] == "agendado"
    ][colunas_stats].drop_duplicates()

    # jogos disputados na temporada atual, um por partida encerrada
    jogos_disputados = (
        partidas_finalizadas_df.groupby("time").size().rename("jogos").reset_index()
    )
    times_stats = times_stats.merge(jogos_disputados, on="time", how="left")
    times_stats["jogos"] = times_stats["jogos"].fillna(0).astype(int)

    tabela_evolucao_df = tabela_evolucao_df.merge(times_stats, on="time", how="left")
    tabela_evolucao_df = tabela_evolucao_df.drop(columns=["vitoria"])

    # garante tipos inteiros nas colunas de rodada/posição para o dashboard
    tabela_evolucao_df["rodada_num"] = tabela_evolucao_df["rodada_num"].astype(int)
    tabela_evolucao_df["posicao"] = tabela_evolucao_df["posicao"].astype(int)
    partidas_df["rodada_num"] = partidas_df["rodada_num"].astype(int)

    return tabela_evolucao_df, partidas_df, times_stats, chances_df


def atualizar(
    n_splits: int = N_SPLITS_MCCV,
    n_jobs: int = N_JOBS,
    executar_loso: bool = EXECUTAR_LOSO,
    executar_mccv: bool = EXECUTAR_MCCV,
) -> None:
    """Executa o pipeline completo e grava os arquivos consumidos pelo dashboard."""
    inicio = time.time()
    partidas_model_final_df, partidas_finalizadas_df = preparar_dados()
    tabela_evolucao_df, partidas_df, times_stats, chances_df = treinar_e_prever(
        partidas_model_final_df,
        partidas_finalizadas_df,
        n_splits=n_splits,
        n_jobs=n_jobs,
        executar_loso=executar_loso,
        executar_mccv=executar_mccv,
    )

    tabela_evolucao_df.to_csv(ARQUIVO_TABELA_EVOLUCAO, index=False)
    partidas_df.to_csv(ARQUIVO_PARTIDAS, index=False)
    times_stats.to_csv(ARQUIVO_TIMES_STATS, index=False)
    if not chances_df.empty:
        chances_df.to_csv(ARQUIVO_CHANCES, index=False)

    logger.info("Arquivos atualizados em %s (%.1fs)", DIR_APP, time.time() - inicio)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atualiza as previsões do Brasileirão."
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=N_SPLITS_MCCV,
        help=f"número de splits da validação Monte Carlo (padrão: {N_SPLITS_MCCV})",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help=f"núcleos usados na validação Monte Carlo (padrão: {N_JOBS})",
    )
    parser.add_argument(
        "--loso",
        dest="executar_loso",
        action="store_true",
        default=EXECUTAR_LOSO,
        help=f"executa a validação Leave-One-Season-Out (LOSO) (padrão: {EXECUTAR_LOSO})",
    )
    parser.add_argument(
        "--sem-loso",
        dest="executar_loso",
        action="store_false",
        help="não executa a validação Leave-One-Season-Out (LOSO)",
    )
    parser.add_argument(
        "--mccv",
        dest="executar_mccv",
        action="store_true",
        default=EXECUTAR_MCCV,
        help=f"executa a validação Monte Carlo (MCCV) (padrão: {EXECUTAR_MCCV})",
    )
    parser.add_argument(
        "--sem-mccv",
        dest="executar_mccv",
        action="store_false",
        help="não executa a validação Monte Carlo (MCCV)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="exibe logs de depuração"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    atualizar(
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
        executar_loso=args.executar_loso,
        executar_mccv=args.executar_mccv,
    )


if __name__ == "__main__":
    main()
