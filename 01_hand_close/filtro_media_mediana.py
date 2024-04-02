import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Carregando Arquivo: 2022-05-05_14-55-25.csv
arquivo_ler = r"C:\Users\GUILHERME\Documents\3_PERÍODO\IC_Daniel\Arquivos_Mao_Robotica\01_hand_close\2022-05-05_14-55-25.csv"
base = pd.read_csv(arquivo_ler, delimiter=";")

# Filtrar apenas os gestos de interesse (hand_open e hand_flex_curl)
gestos_interesse = ["hand_open", "hand_flex_curl"]
base = base[base["gesture"].isin(gestos_interesse)]

# Remover registros duplicados, se houver
base.drop_duplicates(inplace=True)

# Remover registros com valores ausentes (NaN), se houver
base.dropna(inplace=True)

# Filtragem usando Filtro Médio (Moving Average Filter)
def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

# Filtragem usando Filtro de Mediana (Median Filter)
def median_filter(data, window_size):
    return data.rolling(window=window_size, min_periods=1).median()

# Aplicar a filtragem em cada canal usando o Filtro Médio
window_size_ma = 3
channels = ["ch0", "ch1", "ch2", "ch3"]
for channel in channels:
    base[channel] = moving_average(base[channel], window_size_ma)
    

# Aplicar a filtragem em cada canal usando o Filtro de Mediana
window_size_median = 3
channels = ["ch0", "ch1", "ch2", "ch3"]
for channel in channels:
    base[channel] = median_filter(base[channel], window_size_median)
    
# Função para aplicar o filtro de máximo em cada canal
def max_filter(data, window_size):
    return data.rolling(window=window_size, min_periods=1).max()

# Tamanho da janela para o filtro de máximo
window_size_max = 3
channels = ["ch0", "ch1", "ch2", "ch3"]

# Filtro de Módulo
#base[channels] = base[channels].abs()

# Aplicar o filtro de máximo em cada canal
for channel in channels:
    base[channel] = max_filter(base[channel], window_size_max)

# Normalização usando Min-Max Scaling
scaler_minmax = MinMaxScaler(feature_range=(0, 1))  # Definindo o intervalo entre -1 e 1
base[channels] = scaler_minmax.fit_transform(base[channels])

# Salvar os dados pré-processados em um novo arquivo CSV
arquivo_salvar = r"C:\Users\GUILHERME\Documents\3_PERÍODO\IC_Daniel\Arquivos_Mao_Robotica\01_hand_close\2022-05-05_14-55-25_filtrado_normalizado.csv"
base.to_csv(arquivo_salvar, index=False)

# Plotando o Gráfico
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(base.index, base['ch0'], label='Canal 0')
ax.plot(base.index, base['ch1'], label='Canal 1')
ax.plot(base.index, base['ch2'], label='Canal 2')
ax.plot(base.index, base['ch3'], label='Canal 3')

ax.set_xlabel('Amostras')
ax.set_ylabel('Valores')
ax.set_title('Gráfico dos canais Filtrados e Normalizados: Arquivo 2022-05-05_14-55-25.csv')
ax.legend()

plt.show()


# Filtrando os momentos em que 'gesture' é igual a 'hand_open' ou 'hand_flex_curl'
base_open_flex_curl = base.loc[(base['gesture'] == 'hand_open') | (base['gesture'] == 'hand_flex_curl')]

# Plotando o gráfico dos canais para os momentos 'hand_open' e 'hand_flex_curl'
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(base_open_flex_curl.index, base_open_flex_curl['ch0'], label='Canal ch0')
#ax.plot(base_open_flex_curl.index, base_open_flex_curl['ch1'], label='Canal ch1')
#ax.plot(base_open_flex_curl.index, base_open_flex_curl['ch2'], label='Canal ch2')
#ax.plot(base_open_flex_curl.index, base_open_flex_curl['ch3'], label='Canal ch3')

ax.set_xlabel('Amostras')
ax.set_ylabel('Valores')
ax.set_title('Gráfico dos canais Filtrados: Momentos hand_open e hand_flex_curl')
ax.legend()

plt.show()
