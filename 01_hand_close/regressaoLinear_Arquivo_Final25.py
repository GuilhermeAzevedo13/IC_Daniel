import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

# Carregando Arquivo: 2022-05-05_14-55-25.csv
arquivo_ler = r"C:\Users\GUILHERME\Documents\3_PERÍODO\IC_Daniel\Arquivos_Mao_Robotica\01_hand_close\2022-05-05_14-55-25_filtrado_normalizado.csv"
base = pd.read_csv(arquivo_ler, delimiter=";")
print(base.head())

# Substituindo "hand_open" por 1 e "hand_flex_curl" por 0
base['gesture'] = base['gesture'].replace({'hand_open': 1, 'hand_flex_curl': 0})

# Filtrando os momentos em que 'gesture' é igual a 1 (hand_open) ou 0 (hand_flex_curl)
base_open_flex_curl = base.loc[(base['gesture'] == 1) | (base['gesture'] == 0)]

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

#plt.show()

#Escalonando os valores da coluna ch0
x = base['ch0'].values.reshape(-1, 1)
print(x.shape) # Resultado: (120096, 1)

y = base['gesture'].values

# Transformando "hand_open" em 1 e "hand_flex_curl" em 0
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = y_encoded.reshape(-1, 1)
print(y_encoded.shape) # Resultado: (120096, 1)

# Transformar em array os dados
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

scaler_y = StandardScaler()
y_encoded = scaler_y.fit_transform(y_encoded)

print(x) # Os 120.096 valores do ch0
print("\n")
print(y_encoded) # Os 120.096 valores de hand_open e hand_curl
print("\n")

plt.scatter(x,y)
#plt.show()

#formula da regressao linear simples
# y = b0 + b1 * x

np.random.seed(1)
print(np.random.rand(2)) # Gerar 2 valores aleatorios para treinar a Padronização
# valores gerados: [0.417022   0.72032449]

b0 = tf.Variable(0.41) # Devemos inicializar com valores aleatorios para depois melhorar os parametros
b1 = tf.Variable(0.72)

#PlaceHolder
batch_size = 32 #Serve para informar o tamanho do batch para atualizar no sistema de carregamento de dados
xph = tf.placeholder(tf.float32, [batch_size, 1]) # Vai de 32 em 32 ate chegar em 120096
yph = tf.placeholder(tf.float32, [batch_size, 1])

y_modelo = b0 + b1 * xph
erro = tf.losses.mean_squared_error(yph, y_modelo) # Melhores valores de 1 e 0

otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

treinamento = otimizador.minimize(erro) # Vou minimizar os erros de 0 e 1

init = tf.global_variables_initializer() #Inicializar as Variaveis b0 e b1

#Treinamento do Aprendizado
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        indices = np.random.randint(len(x), size = batch_size) #sortear os valores entre 0 e 120096, e buscar entre 32 em 32
        feed = {xph: x[indices], yph: y_encoded[indices]} # preencher os place Holders // Formato de Dicionario
        sess.run(treinamento, feed_dict = feed)
    b0_final, b1_final = sess.run([b0, b1])
    
print(b0_final)
print("\n")
print(b1_final)
print("\n")

# Visualização da regressão linear
x_vis = np.linspace(min(x), max(x), 100)  # Gerar 100 pontos igualmente espaçados no eixo x
y_vis = b0_final + b1_final * x_vis  # Calcula os valores de y correspondentes à regressão

plt.scatter(x, y_encoded)  # Plota os pontos originais
plt.plot(x_vis, y_vis, 'r-', label='Regressão Linear')  # Plota a regressão linear em vermelho
plt.xlabel('ch0 (Valores escalonados)')
plt.ylabel('gesture (Valores escalonados)')
plt.title('Regressão Linear para hand_open e hand_flex_curl')
plt.legend()
plt.show()

previsoes = b0_final + b1_final * x
print(previsoes)

plt.plot(x, y_encoded, 'o')
plt.plot(x, previsoes, color = 'red')
plt.show()

y1 = scaler_y.inverse_transform(y_encoded)
previsoes1 = scaler_y.inverse_transform(previsoes)

print(y1)
print("\n")
print(previsoes1)

mae = mean_absolute_error(y1, previsoes1)
print("\n")
print(mae)

