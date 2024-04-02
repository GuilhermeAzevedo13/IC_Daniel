import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Carregando Arquivo: 2022-05-05_14-55-25.csv
arquivo_ler = r"C:\Users\GUILHERME\Documents\3_PERÍODO\IC_Daniel\Arquivos_Mao_Robotica\01_hand_close\2022-05-05_14-55-25.csv"
base = pd.read_csv(arquivo_ler, delimiter=";")

print(base.head())

# Filtrar apenas os gestos de interesse (hand_open e hand_flex_curl)
gestos_interesse = ["hand_open", "hand_flex_curl"]
base = base[base["gesture"].isin(gestos_interesse)]

# Remover registros duplicados, se houver
base.drop_duplicates(inplace=True)

# Remover registros com valores ausentes (NaN), se houver
base.dropna(inplace=True)

# Filtragem usando Filtro de Mediana (Median Filter)
def median_filter(data, window_size):
    return data.rolling(window=window_size, min_periods=1).median()

# Aplicar a filtragem em cada canal usando o Filtro de Mediana
window_size_median = 3
channels = ["ch0", "ch1", "ch2", "ch3"]
for channel in channels:
    base[channel] = median_filter(base[channel], window_size_median)
    
# Remoção de Outliers usando a regra do IQR
def remove_outliers_iqr(data, column, threshold=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Aplicar a remoção de outliers em cada canal
channels = ["ch0", "ch1", "ch2", "ch3"]
for channel in channels:
    base = remove_outliers_iqr(base, channel)

# Normalização usando Min-Max Scaling
scaler_minmax = MinMaxScaler()
base[channels] = scaler_minmax.fit_transform(base[channels])

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
