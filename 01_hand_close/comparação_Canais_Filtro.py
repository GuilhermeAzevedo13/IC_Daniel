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

# Aplicar a filtragem em cada canal usando o Filtro de Mediana
window_size_median = 3
channels = ["ch0", "ch1", "ch2", "ch3"]
for channel in channels:
    base[channel] = median_filter(base[channel], window_size_median)

# Aplicar a filtragem em cada canal usando o Filtro Médio
window_size_ma = 3
channels = ["ch0", "ch1", "ch2", "ch3"]
for channel in channels:
    base[channel] = moving_average(base[channel], window_size_ma)
    
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

# Plotando os gráficos
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Gráfico dos canais filtrados e normalizados
axes[0, 0].plot(base.index, base['ch0'], label='Canal ch0 (Filtrado e Normalizado)')
axes[0, 0].set_xlabel('Amostras')
axes[0, 0].set_ylabel('Valores')
axes[0, 0].set_title('Gráfico do Canal ch0 (Filtrado e Normalizado)')
axes[0, 0].legend()

# Gráfico dos canais filtrados e normalizados
axes[0, 1].plot(base.index, base['ch1'], label='Canal ch1 (Filtrado e Normalizado)')
axes[0, 1].set_xlabel('Amostras')
axes[0, 1].set_ylabel('Valores')
axes[0, 1].set_title('Gráfico do Canal ch1 (Filtrado e Normalizado)')
axes[0, 1].legend()

# Gráfico dos canais filtrados e normalizados
axes[1, 0].plot(base.index, base['ch2'], label='Canal ch2 (Filtrado e Normalizado)')
axes[1, 0].set_xlabel('Amostras')
axes[1, 0].set_ylabel('Valores')
axes[1, 0].set_title('Gráfico do Canal ch2 (Filtrado e Normalizado)')
axes[1, 0].legend()

# Gráfico dos canais filtrados e normalizados
axes[1, 1].plot(base.index, base['ch3'], label='Canal ch3 (Filtrado e Normalizado)')
axes[1, 1].set_xlabel('Amostras')
axes[1, 1].set_ylabel('Valores')
axes[1, 1].set_title('Gráfico do Canal ch3 (Filtrado e Normalizado)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
