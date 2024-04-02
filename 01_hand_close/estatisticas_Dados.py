import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Carregando Arquivo: 2022-05-05_14-55-25.csv
arquivo_ler = r"C:\Users\GUILHERME\Documents\3_PERÍODO\IC_Daniel\Arquivos_Mao_Robotica\01_hand_close\2022-05-05_14-55-25.csv"
base = pd.read_csv(arquivo_ler, delimiter=";")

# Estatísticas descritivas para os dados do gesto "hand_open"
desc_stats_hand_open = base[base['gesture'] == 'hand_open'].describe()

# Estatísticas descritivas para os dados do gesto "hand_flex_curl"
desc_stats_hand_flex_curl = base[base['gesture'] == 'hand_flex_curl'].describe()

# Imprima as estatísticas descritivas para ambos os gestos
print("Estatísticas descritivas para o gesto 'hand_open':")
print(desc_stats_hand_open)

print("\nEstatísticas descritivas para o gesto 'hand_flex_curl':")
print(desc_stats_hand_flex_curl)

# Plote gráficos para visualizar as estatísticas descritivas
# Gráfico de barras para a média de cada canal em "hand_open" e "hand_flex_curl"
mean_stats = pd.concat([desc_stats_hand_open.loc['mean'], desc_stats_hand_flex_curl.loc['mean']], axis=1)
mean_stats.columns = ['hand_open', 'hand_flex_curl']
mean_stats.plot(kind='bar', figsize=(10, 6))
plt.title('Média por Canal')
plt.xlabel('Canais (ch0, ch1, ch2, ch3)')
plt.ylabel('Média')
plt.show()

# Gráfico de boxplot para visualizar a distribuição dos valores de cada canal em ambos os gestos
boxplot_data = base.drop('gesture', axis=1)
boxplot_data.boxplot(figsize=(10, 6))
plt.title('Distribuição por Canal e Gesto')
plt.xlabel('Canais (ch0, ch1, ch2, ch3)')
plt.ylabel('Leituras')
plt.show()

# Grafico de Dispersão 

# Escolha os canais que você deseja comparar (ch0 e ch1)
channel_x = 'ch0'
channel_y = 'ch1'

# Separe os dados para cada gesto
data_hand_open = base[base['gesture'] == 'hand_open']
data_hand_flex_curl = base[base['gesture'] == 'hand_flex_curl']

# Plote o gráfico de dispersão
plt.figure(figsize=(8, 6))
plt.scatter(data_hand_open[channel_x], data_hand_open[channel_y], label='hand_open', alpha=0.7)
plt.scatter(data_hand_flex_curl[channel_x], data_hand_flex_curl[channel_y], label='hand_flex_curl', alpha=0.7)

# Personalize o gráfico
plt.title(f'Gráfico de Dispersão: {channel_x} vs {channel_y}')
plt.xlabel(channel_x)
plt.ylabel(channel_y)
plt.legend()
plt.grid(True)

# Mostrar o gráfico
plt.show()

# Histograma 
# Escolha o canal que você deseja comparar (ch0)
channel = 'ch0'

# Separe os dados para cada gesto
data_hand_open = base[base['gesture'] == 'hand_open']
data_hand_flex_curl = base[base['gesture'] == 'hand_flex_curl']

# Plote o histograma
plt.figure(figsize=(8, 6))
plt.hist(data_hand_open[channel], bins=20, alpha=0.7, label='hand_open', density=True)
plt.hist(data_hand_flex_curl[channel], bins=20, alpha=0.7, label='hand_flex_curl', density=True)

# Personalize o gráfico
plt.title(f'Histograma do Canal {channel}')
plt.xlabel(channel)
plt.ylabel('Frequência Relativa')
plt.legend()
plt.grid(True)

# Mostrar o gráfico
plt.show()

# Separe os dados para cada gesto
data_hand_open = base[base['gesture'] == 'hand_open']
data_hand_flex_curl = base[base['gesture'] == 'hand_flex_curl']

# Escolha os canais que você deseja analisar (ch0, ch1, ch2, ch3)
channels = ['ch0', 'ch1', 'ch2', 'ch3']

# Análise de correlação para o gesto "hand_open"
correlation_hand_open = data_hand_open[channels].corr()

# Análise de correlação para o gesto "hand_flex_curl"
correlation_hand_flex_curl = data_hand_flex_curl[channels].corr()

# Plote as matrizes de correlação em um heatmap
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
sns.heatmap(correlation_hand_open, annot=True, cmap='coolwarm', center=0)
plt.title('Correlação - hand_open')
plt.xlabel('Canais')
plt.ylabel('Canais')

plt.subplot(1, 2, 2)
sns.heatmap(correlation_hand_flex_curl, annot=True, cmap='coolwarm', center=0)
plt.title('Correlação - hand_flex_curl')
plt.xlabel('Canais')
plt.ylabel('Canais')

plt.tight_layout()
plt.show()