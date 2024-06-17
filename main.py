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

df_confrontos = tables[7]
df_pos_rodada = tables[8]
