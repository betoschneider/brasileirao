import pandas as pd
import requests
from io import StringIO

# URL da tabela de classificação
url = "https://pt.wikipedia.org/wiki/Campeonato_Brasileiro_de_Futebol_de_2024_-_S%C3%A9rie_A"

# Fazendo a requisição para o site
response = requests.get(url)
html_data = response.text

# Envolva a string HTML em um objeto StringIO
html_string_io = StringIO(html_data)

# Use o objeto StringIO com a função read_html
tables = pd.read_html(html_string_io)

# #Adicionar classificação por aproveitamento
df_classificacao = tables[6]
df_classificacao.drop(columns=['Classificação ou descenso'], inplace=True)

df_classificacao['Aprov'] = round((df_classificacao['V'] * 3 + df_classificacao['E']) / (df_classificacao['J'] * 3) * 100, 2)
df_classificacao.sort_values(['Aprov', 'GP', 'SG', 'Pos'], ascending=[False, False, False, True], inplace=True)

df_classificacao = df_classificacao.reset_index(drop=True).reset_index()

df_classificacao['Pos Aprov'] = df_classificacao['index'] + 1
df_classificacao.drop(columns=['index'], inplace=True)
df_classificacao.rename(columns={'Equipevde': 'Equipe'}, inplace=True)

print(df_classificacao)

# resultado dos confrontos
df_confrontos = tables[7]
li_clube_sigla = df_confrontos.columns[1:]
li_clube_nome = df_confrontos['Casa \ Fora'].to_list()

# criando dicionario sigla-nome
dic_clube = {}
for i in range(len(li_clube_sigla)):
    dic_clube[li_clube_sigla[i]] = li_clube_nome[i]

# transformando df_confrontos
li_confronto = []
for i in range(len(li_clube_nome)):
    for clube in li_clube_sigla:
        li_confronto.append([li_clube_nome[i], df_confrontos[clube][i], dic_clube[clube]])

df_confrontos = pd.DataFrame(li_confronto, columns=['mandante', 'placar', 'visitante'])
df_confrontos = df_confrontos[(df_confrontos['placar']!='—') & (df_confrontos['placar']!='a')].dropna()

df_confrontos['gols_mandante'] = df_confrontos.apply(lambda x: x['placar'].split('–')[0], axis=1)
df_confrontos['gols_visitante'] = df_confrontos.apply(lambda x: x['placar'].split('–')[1], axis=1)

# posição de clube por rodada
df_pos_rodada = tables[8].dropna()

print(df_pos_rodada)

li_posicao = []
for rodada in df_pos_rodada[df_pos_rodada.columns[0]][:-1]:
    indice = int(rodada.replace('ª', '')) - 1
    for clube in li_clube_sigla:
        li_posicao.append([dic_clube[clube], indice +1, df_pos_rodada[clube][indice]])

df_pos_rodada = pd.DataFrame(li_posicao, columns=['clube', 'rodada', 'posicao'])
print(df_pos_rodada)