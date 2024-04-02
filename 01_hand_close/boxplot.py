import pandas as pd
import matplotlib.pyplot as plt

# Carregando Arquivo: 2022-05-05_14-55-25.csv
arquivo_ler = r"C:\Users\GUILHERME\Documents\3_PERÍODO\IC_Daniel\Arquivos_Mao_Robotica\01_hand_close\2022-05-05_14-55-25.csv"
base = pd.read_csv(arquivo_ler, delimiter=";")

# Separando os dados por gesto
hand_open_data = base[base['gesture'] == 'hand_open'].drop('gesture', axis=1).astype(int)
hand_flex_curl_data = base[base['gesture'] == 'hand_flex_curl'].drop('gesture', axis=1).astype(int)

# Criando o gráfico de boxplot com cores diferentes para os gestos
plt.figure(figsize=(10, 6))
plt.boxplot([hand_open_data['ch0'], hand_flex_curl_data['ch0']], labels=['Mão Aberta', 'Mão Fechada'])
plt.title('Distribuição por Canal e Gesto')
plt.xlabel('Gesto')
plt.ylabel('Leituras')
plt.show()
