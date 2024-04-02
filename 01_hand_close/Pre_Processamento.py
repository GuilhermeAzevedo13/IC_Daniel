import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
# Com essa normalização, os valores dos canais estarão no intervalo [0, 1], 
# onde 0 representa a mão fechada (gesto "hand_flex_curl") e 1 representa a mão aberta 
# (gesto "hand_open").Portanto, quando você plotar o gráfico com os valores normalizados, 
# os canais estarão na escala [0, 1], mostrando como os valores variam entre mão fechada e 
# mão aberta ao longo das amostras.

# Padronização usando Z-score normalization
#scaler_zscore = StandardScaler()
#base[channels] = scaler_zscore.fit_transform(base[channels])

# Salvar os dados pré-processados em um novo arquivo CSV
arquivo_salvar = r"C:\\Users\\GUILHERME\\Documents\\3_PERÍODO\\IC_Daniel\\Arquivos_Mao_Robotica\\01_hand_close\\2022-05-05_14-55-25_preprocessado.csv"
base.to_csv(arquivo_salvar, index=False)

# Plotando o Gráfico
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(base.index, base['ch0'], label='Canal 0')
ax.plot(base.index, base['ch1'], label='Canal 1')
ax.plot(base.index, base['ch2'], label='Canal 2')
ax.plot(base.index, base['ch3'], label='Canal 3')

ax.set_xlabel('Amostras')
ax.set_ylabel('Valores')
ax.set_title('Gráfico dos canais Filtrados: Arquivo 2022-05-05_14-55-25.csv')
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


