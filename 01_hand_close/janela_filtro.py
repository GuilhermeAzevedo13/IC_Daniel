import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
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

# Normalização usando Min-Max Scaling
scaler_minmax = MinMaxScaler()
channels = ["ch0", "ch1", "ch2", "ch3"]
base[channels] = scaler_minmax.fit_transform(base[channels])

# Calculando o MSE para diferentes tamanhos de janela do Filtro Médio
mse_ma = []
window_sizes_ma = range(3, 21)
for window_size_ma in window_sizes_ma:
    base_ma = base.copy()
    for channel in channels:
        base_ma[channel] = moving_average(base_ma[channel], window_size_ma)
    mse_ma.append(mean_squared_error(base[channels], base_ma[channels]))

# Calculando o MSE para diferentes tamanhos de janela do Filtro de Mediana
mse_median = []
window_sizes_median = range(3, 11)
for window_size_median in window_sizes_median:
    base_median = base.copy()
    for channel in channels:
        base_median[channel] = median_filter(base_median[channel], window_size_median)
    mse_median.append(mean_squared_error(base[channels], base_median[channels]))

# Plotando o MSE para diferentes tamanhos de janela
plt.figure(figsize=(10, 6))
plt.plot(window_sizes_ma, mse_ma, label='MSE Filtro Médio')
plt.plot(window_sizes_median, mse_median, label='MSE Filtro de Mediana')
plt.xlabel('Tamanho da Janela')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Avaliação Quantitativa dos Filtros')
plt.legend()
plt.show()

# Obtendo os melhores tamanhos de janela para cada filtro
best_window_size_ma = window_sizes_ma[np.argmin(mse_ma)]
best_window_size_median = window_sizes_median[np.argmin(mse_median)]

print("Melhor tamanho de janela para o Filtro Médio:", best_window_size_ma)
print("Melhor tamanho de janela para o Filtro de Mediana:", best_window_size_median)
