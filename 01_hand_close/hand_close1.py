import pandas as pd
import matplotlib.pyplot as plt

# Carregando Arquivo: 2022-05-05_14-55-25.csv
arquivo_ler = r"C:\Users\GUILHERME\Documents\3_PERÍODO\IC_Daniel\Arquivos_Mao_Robotica\01_hand_close\2022-05-05_14-55-25.csv"
base = pd.read_csv(arquivo_ler, delimiter=";")
print(base.head())

#hand_close = pd.iloc(pd["gesture"] == "hand_curl")

# Plotando o Gráfico
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(base.index, base['ch0'], label='Canal 0')
ax.plot(base.index, base['ch1'], label='Canal 1')
ax.plot(base.index, base['ch2'], label='Canal 2')
ax.plot(base.index, base['ch3'], label='Canal 3')

ax.set_xlabel('Amostras')
ax.set_ylabel('Valores')
ax.set_title('Gráfico dos canais: Arquivo 2022-05-05_14-55-25.csv')
ax.legend()

plt.show()

# Filtrando os momentos em que 'gesture' é igual a 'hand_open' ou 'hand_flex_curl'
base_open_flex_curl = base.loc[(base['gesture'] == 'hand_open') | (base['gesture'] == 'hand_flex_curl')]

# Plotando o gráfico dos canais para os momentos 'hand_open' e 'hand_flex_curl'
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(base_open_flex_curl.index, base_open_flex_curl['ch0'], label='Canal 0')
#ax.plot(base_open_flex_curl.index, base_open_flex_curl['ch1'], label='Canal 1')
#ax.plot(base_open_flex_curl.index, base_open_flex_curl['ch2'], label='Canal 2')
#ax.plot(base_open_flex_curl.index, base_open_flex_curl['ch3'], label='Canal 3')

ax.set_xlabel('Amostras')
ax.set_ylabel('Valores')
ax.set_title('Gráfico dos canais: Momentos hand_open e hand_flex_curl')
ax.legend()

plt.show()