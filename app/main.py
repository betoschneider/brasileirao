# Importando as bibliotecas
import requests
import urllib3
import pandas as pd
import numpy as np

# Ignorar warnings de certificado SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

######################################
# Captura e enriquecimento dos dados #
######################################

def tratar():

    #######################
    # Dados da competição #
    #######################

    # Requisição dos dados da API
    url = "https://service.ig.com.br/football_ig/campeonatos/10/fases/768"
    response = requests.get(url, verify=False)
    data = response.json()

    # Dados da edição
    edicao_data = {}
    if "edicao" in data:
        for k, v in data["edicao"].items():
            edicao_data[f"edicao.{k}"] = v

    # Tabela
    tabela_data = []
    if "tabela" in data:
        for posicao in data["tabela"]:
            registro = dict(posicao)
            registro.update(edicao_data)
            tabela_data.append(registro)

    tabela_df = pd.json_normalize(tabela_data)

    # Recalcula aproveitamento
    tabela_df['aproveitamento'] = round((tabela_df['vitorias'] * 3 + tabela_df['empates']) / (tabela_df['jogos'] * 3), 4) *100

    # Calcula aproveitamento recente
    def calc_aproveitamento(ultimos):
        if not ultimos:
            return None
        pontos = sum(3 if r == "v" else 1 if r == "e" else 0 for r in ultimos)
        return round(pontos / 15, 4) * 100 if len(ultimos) >= 5 else None

    if "ultimos_jogos" in tabela_df.columns:
        tabela_df["aproveitamento_recente"] = tabela_df["ultimos_jogos"].apply(calc_aproveitamento)

    tabela_df = tabela_df.rename(columns={
            'edicao.edicao_id': 'edicao_id', 
            'edicao.temporada': 'edicao_temporada', 
            'edicao.nome':  'edicao_nome', 
            'edicao.nome_popular': 'edicao_nome_popular',
            'edicao.slug': 'edicao_slug', 
            'time.time_id': 'time_id', 
            'time.nome_popular': 'time_nome_popular', 
            'time.sigla': 'time_sigla', 
            'time.escudo': 'time_escudo'})

    # print(tabela_df.columns)

    colunas_ordenadas = ['edicao_id', 'edicao_temporada',
        'edicao_nome', 'edicao_nome_popular', 'edicao_slug', 'time_id',
        'time_nome_popular', 'time_sigla', 'time_escudo', 'posicao', 'pontos', 'jogos', 'vitorias', 'empates', 'derrotas',
        'gols_pro', 'gols_contra', 'saldo_gols', 'variacao_posicao', 'ultimos_jogos', 'aproveitamento', 'aproveitamento_recente',
        ]
    colunas_exibicao = ['posicao', 'time_sigla', 'jogos', 'pontos', 'vitorias', 'empates', 'derrotas',
        'gols_pro', 'saldo_gols', 'aproveitamento', 'aproveitamento_recente',
        ]
    colunas_exibicao_res = ['posicao', 'time_sigla', 'jogos', 'pontos', 'aproveitamento', 'aproveitamento_recente',
        ]

    tabela_df = tabela_df[colunas_ordenadas]
    tabela_df['data_atualizacao'] = pd.Timestamp.now()

    ############
    # Partidas #
    ############

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
    colunas_descartar_partidas = ["disputa_penalti", "slug", "hora_realizacao",
                                "data_realizacao_iso", "_link", "edicao.nome", "edicao.slug", "estadio",
                                ]
    partidas_df = partidas_df.drop(columns=[c for c in colunas_descartar_partidas if c in partidas_df.columns])

    # rodada int
    def rodada_int(rodada):
        if isinstance(rodada, str):
            # Remove espaços, divide no 'a' e pega a primeira parte
            return rodada.strip().split('a')[0]
        return None  # Caso o valor não seja string

    if "rodada" in partidas_df.columns:
        partidas_df["rodada_num"] = partidas_df["rodada"].apply(rodada_int).astype(int)

    #####################
    # add valor do time #
    #####################
    # arquivo csv com valores dos times
    valor_time_df = pd.read_csv("valor-time-titular.csv")
    partidas_df['edicao.temporada'] = partidas_df['edicao.temporada'].astype('int64')
    # mandante
    partidas_df = partidas_df.merge(valor_time_df, left_on=['time_mandante.nome_popular', 'edicao.temporada'], right_on=['time', 'ano'], how='left')
    partidas_df.drop(columns=['time', 'ano'], inplace=True)
    partidas_df.rename(columns={'time.valor': 'time_mandante.valor'}, inplace=True)
    # visitante
    partidas_df = partidas_df.merge(valor_time_df, left_on=['time_visitante.nome_popular', 'edicao.temporada'], right_on=['time', 'ano'], how='left')
    partidas_df.drop(columns=['time', 'ano'], inplace=True)
    partidas_df.rename(columns={'time.valor': 'time_visitante.valor'}, inplace=True)

    # formata data
    partidas_df["data_realizacao"] = pd.to_datetime(partidas_df["data_realizacao"], format="%d/%m/%Y")

    ###########################
    # campeoanatos anteriores #
    ###########################
    # arquivo csv com dados históricos
    historico_df = pd.read_csv("br-historico-2022-2024.csv")
    historico_df['data_realizacao'] = pd.to_datetime(historico_df['data_realizacao'], unit='D', origin='1899-12-30')

    # add registros antigos ao campeonato atual
    partidas_df = pd.concat([partidas_df, historico_df], ignore_index=True)

    # definindo colunas
    colunas_mandante = ['rodada', 'rodada_num', 'data_realizacao', 'partida_id', 'estadio.estadio_id', 'estadio.nome_popular',
                        'status', 'edicao.edicao_id', 'edicao.temporada', 'edicao.nome_popular', 'campeonato.campeonato_id', 
                        'campeonato.nome', 'campeonato.slug', 'time_mandante.time_id', 'time_mandante.nome_popular',
                        'time_visitante.time_id', 'time_visitante.nome_popular', 'placar_mandante', 'placar_visitante',
                        'time_mandante.valor', 'time_visitante.valor',
                        ]
    colunas_visitante = ['rodada', 'rodada_num', 'data_realizacao', 'partida_id', 'estadio.estadio_id', 'estadio.nome_popular',
                        'status', 'edicao.edicao_id', 'edicao.temporada', 'edicao.nome_popular', 'campeonato.campeonato_id', 
                        'campeonato.nome', 'campeonato.slug', 'time_mandante.time_id', 'time_mandante.nome_popular',
                        'time_visitante.time_id', 'time_visitante.nome_popular', 'placar_visitante', 'placar_mandante',
                        'time_mandante.valor', 'time_visitante.valor',
                        ]

    # modelagem para as projeções
    partidas_model_df = None
    temp = None

    for partida in partidas_df['partida_id']:
        # mandante
        temp = partidas_df[(partidas_df['partida_id'] == partida)]
        temp = temp[colunas_mandante]
        temp['mandante'] = 1
        temp.rename(columns={
            'time_mandante.time_id': 'time_id',
            'time_mandante.nome_popular': 'time',
            'time_visitante.time_id': 'adversario_id',
            'time_visitante.nome_popular': 'adversario',
            'time_mandante.valor': 'valor_time',
            'time_visitante.valor': 'valor_adversario',
            'placar_mandante': 'gols_time',
            'placar_visitante': 'gols_adversario',
        }, inplace=True)
        partidas_model_df = pd.concat([partidas_model_df, temp], ignore_index=True)

        # visitante
        temp = partidas_df[(partidas_df['partida_id'] == partida)]
        temp = temp[colunas_visitante]
        temp['mandante'] = 0
        temp.rename(columns={
            'time_mandante.time_id': 'adversario_id',
            'time_mandante.nome_popular': 'adversario',
            'time_visitante.time_id': 'time_id',
            'time_visitante.nome_popular': 'time',
            'time_mandante.valor': 'valor_adversario',
            'time_visitante.valor': 'valor_time',
            'placar_visitante': 'gols_time',
            'placar_mandante': 'gols_adversario',
        }, inplace=True)
        partidas_model_df = pd.concat([partidas_model_df, temp], ignore_index=True)

    # classifica df
    partidas_model_df.sort_values(by=['data_realizacao', 'rodada_num', 'partida_id', 'mandante'], ascending=[True, True, True, False], inplace=True)

    partidas_agendadas_df = partidas_model_df[partidas_model_df['status'] != 'finalizado']
    partidas_model_df = partidas_model_df[partidas_model_df['status'] == 'finalizado']       

    # add ano-rodada
    partidas_model_df['ano-rodada'] = partidas_model_df['edicao.temporada'].astype(str) + "." + partidas_model_df['rodada_num'].astype(str)

    #################
    # média de gols #
    #################
    # calcula média de gols marcados e sofridos por time até a rodada anterior
    media_gols_df = None
    for rodada in partidas_model_df['ano-rodada'].drop_duplicates():
        temp = partidas_model_df[(partidas_model_df['rodada_num'] < int(rodada.split(".")[1])) &
                                (partidas_model_df['edicao.temporada'] == int(rodada.split(".")[0]))]
        temp = temp[['time', 'gols_time', 'gols_adversario']].groupby('time').mean().reset_index()
        temp['ano-rodada'] = rodada
        media_gols_df = pd.concat([media_gols_df, temp], ignore_index=True)

    media_gols_df.rename(columns={'gols_time': 'media_gols_marcados', 'gols_adversario': 'media_gols_sofridos'}, inplace=True)

    # add média time
    partidas_model_df = partidas_model_df.merge(media_gols_df, on=['ano-rodada', 'time'], how='left')

    # add média adversário
    media_gols_df.rename(columns={'time': 'adversario', 'media_gols_marcados': 'media_gols_marcados_adversario', 'media_gols_sofridos': 'media_gols_sofridos_adversario'}, inplace=True)
    partidas_model_df = partidas_model_df.merge(media_gols_df, on=['ano-rodada', 'adversario'], how='left')


    #######################
    # pontos conquistados #
    #######################
    # calcula pontos conquistados por time até a rodada anterior
    # vitória = 3 pontos, empate = 1 ponto, derrota = 0 pontos
    partidas_model_df['pontos'] = partidas_model_df.apply(
        lambda x: 3 if x['gols_time'] > x['gols_adversario'] else (1 if x['gols_time'] == x['gols_adversario'] else (0 if x['status'] != 'agendado' else np.nan)),
        axis=1
    )

    pontos_df = None
    for rodada in partidas_model_df['ano-rodada'].drop_duplicates():
        temp = partidas_model_df[(partidas_model_df['rodada_num'] < int(rodada.split(".")[1])) &
                                (partidas_model_df['edicao.temporada'] == int(rodada.split(".")[0]))]
        temp = temp[['time', 'pontos']].groupby('time').sum().reset_index()
        temp['ano-rodada'] = rodada
        pontos_df = pd.concat([pontos_df, temp], ignore_index=True)

    # add média time
    pontos_df.rename(
        columns={
            'pontos': 'pontos_time'
        }, 
        inplace=True
    )
    partidas_model_df = partidas_model_df.merge(pontos_df, on=['ano-rodada', 'time'], how='left')

    # add média adversario
    pontos_df.rename(
        columns={
            'pontos_time': 'pontos_adversario',
            'time': 'adversario'
        }, 
        inplace=True
    )
    partidas_model_df = partidas_model_df.merge(pontos_df, on=['ano-rodada', 'adversario'], how='left')


    ##################
    # Aproveitamento #
    ##################
    # cria a coluna "jogos" inicialmente como 0
    partidas_model_df["jogos"] = np.nan

    # conta só partidas já realizadas (por temporada e time)
    partidas_model_df.loc[partidas_model_df["status"] != "agendado", "jogos"] = (
        partidas_model_df.loc[partidas_model_df["status"] != "agendado"]
        .groupby(["edicao.temporada", "time"])
        .cumcount()
    )

    # preenche os "agendados" com o valor anterior do mesmo time na mesma temporada
    partidas_model_df["jogos"] = (
        partidas_model_df.groupby(["edicao.temporada", "time"])["jogos"]
        .ffill()
        .fillna(0)
        .astype(int)
    )

    # pontos acumulados até a rodada (se for ano-rodada já incluído, pode trocar)
    partidas_model_df["pontos_acumulados"] = (
        partidas_model_df.groupby(["edicao.temporada", "time"])["pontos_time"]
        .cumsum()
    )

    # aproveitamento (%)
    partidas_model_df["aproveitamento"] = (
        partidas_model_df["pontos_acumulados"] / (partidas_model_df["jogos"] * 3)
    )

    ###############################################
    # Aproveitamento recente (últimas 5 partidas) #
    ###############################################
    from collections import deque

    window = 5

    # ordena por temporada, time e rodada
    partidas_model_df = partidas_model_df.sort_values(["edicao.temporada", "time", "rodada_num"])

    # listas para armazenar resultados
    pontos_recentes = []
    jogos_validos = []

    # processa temporada + time
    for (temporada, time), g in partidas_model_df.groupby(["edicao.temporada", "time"]):
        last_points = deque()  # pontos das últimas 'window' partidas realizadas
        for idx, row in g.iterrows():
            # calcula soma e quantidade **antes de incluir a rodada atual**
            pontos_recentes.append(sum(last_points))
            jogos_validos.append(len(last_points))
            
            # adiciona os pontos da rodada atual somente se foi realizada
            if row["status"] != "agendado":
                last_points.append(row["pontos"])
            
            # mantém no máximo 'window' jogos realizados
            while len(last_points) > window:
                last_points.popleft()

    # adiciona ao DataFrame
    partidas_model_df["pontos_recentes"] = pontos_recentes
    partidas_model_df["jogos_validos"] = jogos_validos
    partidas_model_df["aproveitamento_recente"] = (
        partidas_model_df["pontos_recentes"] / (partidas_model_df["jogos_validos"] * 3)
    )

    # --- adiciona aproveitamento e aproveitamento recente para adversário ---
    # seleciona as colunas do adversário, incluindo temporada
    adversario_cols = ["edicao.temporada", "rodada_num", "time", "aproveitamento", "aproveitamento_recente"]

    # renomeia as colunas do adversário
    adversario_renomeado = partidas_model_df[adversario_cols].rename(
        columns={
            "time": "adversario",
            "aproveitamento": "aproveitamento_adversario",
            "aproveitamento_recente": "aproveitamento_recente_adversario"
        }
    )

    # merge com o próprio dataframe (mesma temporada, rodada e adversário)
    partidas_model_df = partidas_model_df.merge(
        adversario_renomeado,
        on=["edicao.temporada", "rodada_num", "adversario"],
        how="left"
    )

    #####################
    # Confronto recente #
    #####################
    def calcular_confronto_recente(partidas_model_df, window=5, por_temporada=True):
        df = partidas_model_df.copy().reset_index(drop=True)
        df = df[df['status']=='finalizado']

        # mantém apenas a visão do time
        A = df[["edicao.temporada", "rodada_num", "time", "adversario", "pontos", "status"]].copy()

        # ordena
        if por_temporada:
            sort_cols = ["edicao.temporada", "time", "adversario", "rodada_num"]
            group_cols = ["edicao.temporada", "time", "adversario"]
        else:
            sort_cols = ["time", "adversario", "edicao.temporada", "rodada_num"]
            group_cols = ["time", "adversario"]

        A = A.sort_values(sort_cols).reset_index(drop=True)

        # calcular aproveitamento nos últimos confrontos
        confronto_valores = {}
        for keys, g in A.groupby(group_cols, sort=False):
            last = deque()
            for idx, row in g.iterrows():
                if len(last) > 0:
                    aproveitamento = sum(last) / (len(last) * 3)
                else:
                    aproveitamento = np.nan
                confronto_valores[idx] = aproveitamento

                if row["status"] != "agendado":
                    last.append(int(row["pontos"]))
                if len(last) > window:
                    last.popleft()

        A["confronto_recente"] = A.index.map(confronto_valores)

        # merge de volta ao df original
        out = df.merge(
            A[["edicao.temporada", "rodada_num", "time", "adversario", "confronto_recente"]],
            on=["edicao.temporada", "rodada_num", "time", "adversario"],
            how="left",
        )

        # adversario
        out_adv = out.copy()
        # out_adv.drop(labels='confronto_recente_adversario', inplace=True)
        out_adv = out_adv[
            [
                'edicao.temporada', 
                'rodada_num', 
                'time', 
                'adversario', 
                'confronto_recente', 
            ]
        ].rename(columns={'time': 'adversario',
                'adversario': 'time',
                'confronto_recente': 'confronto_recente_adversario'})
        
        # add confronto recente adversario
        out = out.merge(out_adv, on=['edicao.temporada', 'rodada_num', 'time', 'adversario'], how='left')

        out['confronto_recente'] = out['confronto_recente'].fillna(0)
        out['confronto_recente_adversario'] = out['confronto_recente_adversario'].fillna(0)
        
        return out


    partidas_model_df = calcular_confronto_recente(partidas_model_df, window=5, por_temporada=False)
    # por_temporada=True se quiser reiniciar o histórico a cada temporada

    ##############################################
    # add dados enriquecidos aos jogos agendados #
    ##############################################

    # pega a maior rodada por temporada + time
    df_max_rodada = (
        partidas_model_df
        .groupby(['edicao.temporada', 'time'], as_index=False)['rodada_num']
        .max()
    )

    # reune com o dataframe original para trazer as demais colunas da linha da maior rodada
    df_max_rodada = df_max_rodada.merge(
        partidas_model_df,
        on=['edicao.temporada', 'time', 'rodada_num'],
        how='left'
    ).reset_index(drop=True)

    df_max_rodada = df_max_rodada[['edicao.temporada', 'time', 'media_gols_marcados', 'media_gols_sofridos', 'pontos', 
                                'pontos_time', 'jogos', 'pontos_acumulados', 'aproveitamento', 'pontos_recentes',
                                'jogos_validos', 'aproveitamento_recente', 'confronto_recente']]

    partidas_agendadas_df = partidas_agendadas_df.merge(df_max_rodada, on=['edicao.temporada', 'time'], how='left')

    df_max_rodada.rename(columns={
        'time': 'adversario',
        'media_gols_marcados': 'media_gols_marcados_adversario',
        'media_gols_sofridos': 'media_gols_sofridos_adversario',
        'pontos_time': 'pontos_adversario',
        'aproveitamento': 'aproveitamento_adversario',
        'aproveitamento_recente': 'aproveitamento_recente_adversario',
        'confronto_recente': 'confronto_recente_adversario',
        }, inplace=True)
    df_max_rodada.drop(columns=['pontos', 'jogos', 'pontos_acumulados', 'pontos_recentes', 'jogos_validos',], inplace=True)

    partidas_agendadas_df = partidas_agendadas_df.merge(df_max_rodada, on=['edicao.temporada', 'adversario'], how='left')

    ###########################
    # df para aplicar modelos #
    ###########################
    partidas_model_df = pd.concat([partidas_model_df, partidas_agendadas_df], ignore_index=True)

    colunas_descartar_partidas = ['rodada', 'estadio.nome_popular', 'edicao.edicao_id',
                                'ano-rodada', 'edicao.nome_popular', 'campeonato.campeonato_id',
                                'campeonato.nome', 'campeonato.slug',
                                'jogos', 'pontos_acumulados', "pontos_validos", "pontos_recentes", 
                                "jogos_validos", "pontos_shifted",
                                ]
    partidas_model_final_df = partidas_model_df.drop(columns=[c for c in colunas_descartar_partidas if c in partidas_model_df.columns])

    partidas_finalizadas_df = partidas_model_final_df[(partidas_model_final_df['status'] != 'agendado') &
                                                    #   (partidas_model_final_df['mandante'] == 1) &
                                                    (partidas_model_final_df['edicao.temporada'] == partidas_model_final_df['edicao.temporada'].max())
                                                    ]
    # remove rodada 1 (sem dados anteriores)
    partidas_model_final_df = partidas_model_final_df[partidas_model_final_df["rodada_num"] != 1].reset_index(drop=True)

    return partidas_model_final_df



#############################################################
#                                                           #
#            Treinamento e aplicação dos modelos            #
#                                                           #
#############################################################

# importando as bibliotecas
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')

def executar_modelo(partidas_model_final_df):
    # filtra apenas partidas realizadas
    df = partidas_model_final_df[(partidas_model_final_df["status"] != "agendado")].copy()

    # remove colunas irrelevantes
    exclude_cols = ["confronto_recente_adversario", "estadio.estadio_id", "data_realizacao", "status", "time", "adversario", "partida_id", "time_id", "adversario_id", "gols_time", "gols_adversario"]
    df = df.drop(columns=exclude_cols)


    #####################################
    #                                   #
    #   Leave-One-Season-Out (LOSO)     #
    #                                   #
    #####################################

    # ================================
    # 1️⃣ Features
    # ================================
    features = [
        "mandante", "rodada_num",
        "aproveitamento", "aproveitamento_recente",
        "aproveitamento_adversario", "aproveitamento_recente_adversario",
        "media_gols_marcados", "media_gols_sofridos",
        "media_gols_marcados_adversario", "media_gols_sofridos_adversario",
    ]

    # ================================
    # 2️⃣ Separar jogos realizados e agendados
    # ================================
    df_realizados = partidas_model_final_df[partidas_model_final_df["status"] != "agendado"].copy()
    df_agendadas  = partidas_model_final_df[partidas_model_final_df["status"] == "agendado"].copy()

    X_realizados = df_realizados[features].copy()
    y_realizados = df_realizados["pontos"]
    X_realizados["mandante"] = X_realizados["mandante"].astype(int)

    X_agendadas = df_agendadas[features].copy()
    X_agendadas["mandante"] = X_agendadas["mandante"].astype(int)

    # ================================
    # 3️⃣ Classificadores
    # ================================
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    }

    # ================================
    # 4️⃣ Validação temporal Leave-One-Season-Out
    # ================================
    df_realizados = df_realizados.sort_values(["edicao.temporada", "rodada_num"])
    temporadas = sorted(df_realizados["edicao.temporada"].unique())

    print("=== Validação temporal Leave-One-Season-Out ===")
    for name, clf in classifiers.items():
        accs = []
        for i in range(1, len(temporadas)):
            train_seasons = temporadas[:i]   # até a temporada anterior
            test_season   = temporadas[i]    # temporada alvo

            X_train = df_realizados[df_realizados["edicao.temporada"].isin(train_seasons)][features].copy()
            y_train = df_realizados[df_realizados["edicao.temporada"].isin(train_seasons)]["pontos"]

            X_test  = df_realizados[df_realizados["edicao.temporada"] == test_season][features].copy()
            y_test  = df_realizados[df_realizados["edicao.temporada"] == test_season]["pontos"]

            X_train["mandante"] = X_train["mandante"].astype(int)
            X_test["mandante"]  = X_test["mandante"].astype(int)

            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            accs.append(acc)

        print(f"{name}: média acc={np.mean(accs):.3f} ± {np.std(accs):.3f}")

    # ================================
    # 5️⃣ Projeção futura (treina em todo histórico e prevê 2025+)
    # ================================
    # resumo_agendadas = pd.DataFrame(index=df_agendadas.index)
    loso_resumo_agendadas_df = df_agendadas[["rodada_num", "partida_id", "mandante", "time", "adversario"]].sort_values(by=["rodada_num", "partida_id", "mandante"], ascending=[True, True, False]).reset_index(drop=True)

    for name, clf in classifiers.items():
        clf.fit(X_realizados, y_realizados)  # agora treina no histórico inteiro
        pred = clf.predict(X_agendadas)

        loso_resumo_agendadas_df[f"{name}_pred"] = pred


    ############################################
    #                                          #
    #   Monte Carlo Cross-Validation (MCCV)    #
    #                                          #
    ############################################

    # Features já definidas
    features = [
        "mandante", "rodada_num",
        "aproveitamento", "aproveitamento_recente",
        "aproveitamento_adversario", "aproveitamento_recente_adversario",
        "media_gols_marcados", "media_gols_sofridos",
        "media_gols_marcados_adversario", "media_gols_sofridos_adversario",
    ]

    # Separar dados realizados e agendados
    df_realizados = partidas_model_final_df[partidas_model_final_df["status"] != "agendado"].copy()
    X_realizados = df_realizados[features].copy()
    y_realizados = df_realizados["pontos"]
    X_realizados["mandante"] = X_realizados["mandante"].astype(int)

    df_agendadas = partidas_model_final_df[partidas_model_final_df["status"] == "agendado"].copy()
    X_agendadas = df_agendadas[features].copy()
    X_agendadas["mandante"] = X_agendadas["mandante"].astype(int)

    # Classificadores
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced"),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        # "GradientBoosting": GradientBoostingClassifier(n_estimators=200),
        # "KNN": KNeighborsClassifier(n_neighbors=5),
        # "NaiveBayes": GaussianNB()
    }

    n_splits = 1000
    # resumo_agendadas = pd.DataFrame(index=df_agendadas.index)
    mccv_resumo_agendadas_df = df_agendadas[["rodada_num", "partida_id", "mandante", "time", "adversario"]].sort_values(by=["rodada_num", "partida_id", "mandante"], ascending=[True, True, False]).reset_index(drop=True)

    for name, clf in classifiers.items():
        all_preds = []
        for i in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                X_realizados, y_realizados, test_size=0.2, stratify=y_realizados, random_state=i
            )
            clf.fit(X_train, y_train)
            pred = clf.predict(X_agendadas)
            all_preds.append(pred)
        
        all_preds = np.array(all_preds).T  # n_partidas x n_splits
        modo = [np.bincount(row.astype(int)).argmax() for row in all_preds]
        freq_percent = [ {p: (np.sum(row==p)/n_splits)*100 for p in [0,1,3]} for row in all_preds ]
        
        mccv_resumo_agendadas_df[f"{name}_pred"] = modo
        mccv_resumo_agendadas_df[f"{name}_freq"] = freq_percent

    # Para RandomForest
    mccv_resumo_agendadas_df["RandomForest_pred_freq"] = mccv_resumo_agendadas_df.apply(
        lambda row: row["RandomForest_freq"].get(row["RandomForest_pred"], 0),
        axis=1
    )

    # Se quiser fazer o mesmo para LogisticRegression
    mccv_resumo_agendadas_df["LogisticRegression_pred_freq"] = mccv_resumo_agendadas_df.apply(
        lambda row: row["LogisticRegression_freq"].get(row["LogisticRegression_pred"], 0),
        axis=1
    )

    def gerar_grafico_percentual(freq_dict):
        # freq_dict = {0: xx%, 1: xx%, 3: xx%}
        total_blocos = 10  # define o tamanho do gráfico
        grafico = ""
        for p in [0,1,3]:
            n_blocos = int((freq_dict.get(p,0)/100)*total_blocos)
            grafico += f"{p}:{'█'*n_blocos}{' '*(total_blocos-n_blocos)} "
        return grafico.strip()

    # Aplicando no DataFrame resumo_agendadas
    for name in classifiers.keys():
        mccv_resumo_agendadas_df[f"{name}_grafico"] = mccv_resumo_agendadas_df[f"{name}_freq"].apply(gerar_grafico_percentual)

    # Exibe tabela final compacta
    cols = []
    for name in classifiers.keys():
        cols += [f"{name}_pred", f"{name}_grafico"]


    ############################################
    #                                          #
    #   Adicionando previsões aos resultados   #
    #                                          #
    ############################################

    # LOSO
    modelos = ['RandomForest_pred', 'LogisticRegression_pred']
    tabela_evolucao_df = None
    for modelo in modelos:
        temp_df = pd.concat(
            [
                partidas_finalizadas_df[['rodada_num', 'time', 'pontos']], 
                loso_resumo_agendadas_df[['rodada_num', 'time', modelo]].rename(columns={modelo: 'pontos'})
            ], ignore_index=True)
        temp_df = temp_df.sort_values(by=['time', 'rodada_num'], ascending=[True, True]).reset_index(drop=True)
        temp_df['pontos'] = temp_df.groupby(['time'])["pontos"].cumsum()
        temp_df['modelo'] = modelo.replace('_pred', '_LOSO')
        tabela_evolucao_df = pd.concat([tabela_evolucao_df, temp_df], ignore_index=True)

    # calcula posição por rodada e modelo
    tabela_evolucao_df.sort_values(by=['rodada_num', 'modelo', 'pontos'], ascending=[True, True, False], inplace=True)
    tabela_evolucao_df["posicao"] = (
        tabela_evolucao_df
        .groupby(["rodada_num", "modelo"])["pontos"]
        .rank(method="min", ascending=False)  # menor rank = melhor posição
        .astype(int)
    )

    # MCCV
    modelos = ['RandomForest_pred', 'LogisticRegression_pred']
    # tabela_evolucao_df = None
    for modelo in modelos:
        temp_df = pd.concat(
            [
                partidas_finalizadas_df[['rodada_num', 'time', 'pontos']], 
                mccv_resumo_agendadas_df[['rodada_num', 'time', modelo]].rename(columns={modelo: 'pontos'})
            ], ignore_index=True)
        temp_df = temp_df.sort_values(by=['time', 'rodada_num'], ascending=[True, True]).reset_index(drop=True)
        temp_df['pontos'] = temp_df.groupby(['time'])["pontos"].cumsum()
        temp_df['modelo'] = modelo.replace('_pred', '_MCCV')
        tabela_evolucao_df = pd.concat([tabela_evolucao_df, temp_df], ignore_index=True)

    # calcula posição por rodada e modelo
    tabela_evolucao_df.sort_values(by=['rodada_num', 'modelo', 'pontos'], ascending=[True, True, False], inplace=True)
    tabela_evolucao_df["posicao"] = (
        tabela_evolucao_df
        .groupby(["rodada_num", "modelo"])["pontos"]
        .rank(method="min", ascending=False)  # menor rank = melhor posição
        .astype(int)
    )

    # lista de partidas
    partidas_finalizadas_df = partidas_finalizadas_df[['rodada_num', 'mandante', 'time', 'adversario', 'pontos']]
    partidas_finalizadas_df['status'] = 'Finalizado'

    loso_resumo_agendadas_df['validacao'] = 'LOSO'
    mccv_resumo_agendadas_df['validacao'] = 'MCCV'

    partidas_predicao_df = pd.concat(
        [
            loso_resumo_agendadas_df[['rodada_num', 'mandante', 'time', 'adversario', 'RandomForest_pred', 'LogisticRegression_pred', 'validacao']],
            mccv_resumo_agendadas_df[['rodada_num', 'mandante', 'time', 'adversario', 'RandomForest_pred', 'RandomForest_pred_freq', 'LogisticRegression_pred', 'LogisticRegression_pred_freq', 'validacao']]
        ], ignore_index=True)
    partidas_predicao_df['status'] = 'Previsto'

    modelos = ['RandomForest_pred', 'LogisticRegression_pred']
    partidas_df = None
    for modelo in modelos:
        temp = partidas_predicao_df[['rodada_num', 'mandante', 'time', 'adversario', modelo, f'{modelo}_freq','status', 'validacao']].copy()
        temp.rename(columns={modelo: 'pontos', f'{modelo}_freq': 'frequencia'}, inplace=True)
        temp['modelo'] = temp.apply(lambda x: f'{modelo}'.replace('_pred', '') + ('_MCCV' if x['validacao'] == 'MCCV' else '_LOSO'), axis=1)
        partidas_df = pd.concat([partidas_df, temp], ignore_index=True)

    partidas_df = pd.concat([partidas_df, partidas_finalizadas_df], ignore_index=True)
    partidas_df['resultado'] = partidas_df.apply(lambda x: 'Vitória' if x['pontos'] == 3 else ('Empate' if x['pontos'] == 1 else 'Derrota'), axis=1)
    partidas_df['mandante'] = partidas_df['mandante'].apply(lambda x: 'Sim' if x == 1 else 'Não')
    partidas_df.sort_values(by=['rodada_num', 'time', 'mandante'], ascending=[True, True, False], inplace=True)
    partidas_df = partidas_df[['rodada_num', 'status', 'mandante', 'time', 'adversario', 'resultado', 'frequencia', 'modelo']]

    # add coluna com data de atualização
    tabela_evolucao_df['data_atualizacao'] = pd.Timestamp.now()

    ########################################
    #    Exportando resultados para CSV    #
    ########################################
    tabela_evolucao_df.to_csv("tabela-evolucao-modelos.csv", index=False)
    partidas_df.to_csv("partidas-modelos.csv", index=False)

if __name__ == "__main__":
    df = tratar()
    executar_modelo(df)