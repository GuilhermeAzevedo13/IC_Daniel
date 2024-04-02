from sklearn import datasets
iris = datasets.load_iris()
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import time

caminho_do_arquivo_original = "C:\\Users\\guilh\\OneDrive\\Documents\\3_periodo_ufs\\IC - Daniel\\Treinamento Phyton\\perceptron\\classificacao_binaria_XOR\\normalizado_com_gesture_1_e_0.csv"

df = pd.read_csv(caminho_do_arquivo_original)

data_array = np.array(df)

# Separando as colunas
X = data_array[:, 2:6]  # Colunas 'ch0', 'ch1', 'ch2', 'ch3'
y = data_array[:, 1].reshape(-1, 1)  # Coluna 'gesture'

print("X.shape: {}".format(X.shape))
print("\n")
print("X: (Atributos Previsores)")
print(X)
print("\n")
print("y.shape: {}".format(y.shape))
print("\n")
print("y:")
print(y)
print("\n")

scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
print("X escalonado:")
print(X)
print("\n")

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)
print("y.shape:")
print(y.shape)
y = y.reshape(-1,1)
print("\n")
print("y.reshape:")
print(y.shape)

y = np.array(ct.fit_transform(y), dtype=np.float32)
print("\n")
print("y combinado para a saida ")
print(y)

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.3)

print("X_treinamento.shape:")
print(X_treinamento.shape)

print("X_teste.shape:")
print(X_teste.shape)

neuronios_entrada = X.shape[1]
print("Neuronios_entrada:")
print(neuronios_entrada)

#neuronios_oculta  = np.ceil((X.shape[1] + y.shape[1]) / 2)
neuronios_oculta  = 12
print("Neuronios_oculta:")
print(neuronios_oculta)

neuronios_saida = y.shape[1]
print("Neuronios_saida:")
print(neuronios_saida)

#pesos
W = {'oculta': tf.Variable(tf.random.normal([int(neuronios_entrada), int(neuronios_oculta)])),
     'saida': tf.Variable(tf.random.normal([int(neuronios_oculta), int(neuronios_saida)]))}


bias = {'oculta': tf.Variable(tf.random.normal([int(neuronios_oculta)])),
        'saida': tf.Variable(tf.random.normal([int(neuronios_saida)]))}

xph = tf.compat.v1.placeholder('float', [None, neuronios_entrada])
yph = tf.compat.v1.placeholder('float', [None, neuronios_saida])

def mlp(x, W, bias):
    camada_oculta = tf.add(tf.matmul(x, W['oculta']), bias['oculta'])
    camada_oculta_ativacao = tf.nn.relu(camada_oculta)
    camada_saida = tf.add(tf.matmul(camada_oculta_ativacao, W['saida']), bias['saida'])
    return camada_saida

modelo = mlp(xph, W, bias)

#erro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = modelo, labels = yph)) # Multiclasse
erro = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=modelo, labels=yph)) # Classe Binaria

"""
tf.compat.v1.train.AdamOptimizer: Isso cria uma instância do otimizador Adam. 
O otimizador Adam é uma variação do gradiente descendente estocástico que adapta 
automaticamente as taxas de aprendizado para cada parâmetro da rede com base em suas 
magnitudes dos gradientes anteriores. Isso geralmente ajuda a convergir mais rapidamente 
em problemas de otimização não convexos, como o treinamento de redes neurais.

learning_rate=0.0001 ou 0.001 ou 0.01 ou 0.1: Este é o hiperparâmetro de taxa de aprendizado. 
Ele determina a magnitude dos ajustes feitos nos pesos da rede durante o treinamento. 
Um valor muito alto pode levar a oscilações ou a falta de convergência, 
enquanto um valor muito baixo pode resultar em treinamento lento ou estagnação.

.minimize(erro): Esta função solicita ao otimizador que minimize a função de erro especificada (erro). 
O otimizador então ajustará automaticamente os pesos da rede para tentar minimizar o valor dessa função.
"""
otimizador = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(erro)

batch_size = 128
batch_total = len(X_treinamento) / batch_size
print("Batch_Size: {}".format(batch_size))
print("Tamanho X_treinamento: {}".format(len(X_treinamento)))
print("batch_total:")
print(batch_total)
print("\n")

X_batches = np.array_split(X_treinamento, batch_total)
#print("X_batches:")
#print(X_batches)


# Marcação de tempo inicial
start_time = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoca in range(10000):
        erro_medio = 0.0
        batch_total = int(len(X_treinamento) / batch_size)
        X_batches = np.array_split(X_treinamento, batch_total)
        y_batches = np.array_split(y_treinamento, batch_total)
        for i in range(batch_total):
            X_batch, y_batch = X_batches[i], y_batches[i]
            _, custo = sess.run([otimizador, erro], feed_dict = {xph: X_batch, yph: y_batch})
            erro_medio += (custo / batch_total)
            if epoca % 500 == 0:
                print('Época: ' + str((epoca + 1)) + '\n' + 'erro: ' + str(erro_medio))
    W_final, bias_final = sess.run([W, bias])
    
print("W_final:")
print(W_final)
print("\n")
print("Bias_final:")
print(bias_final)

#previsoes
previsoes_teste = mlp(xph, W_final, bias_final)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    """resuktado 1: Isso calcula as previsões do modelo para o conjunto de teste (X_teste). 
       A variável previsoes_teste é a saída da última camada da rede neural, 
       representando as previsões do modelo para cada exemplo de teste.
    """
    resultado1  = sess.run(previsoes_teste, feed_dict= {xph: X_teste})
    
    """resultado 2: Isso aplica a função softmax às previsões do modelo. 
       A função softmax transforma as saídas em uma distribuição de probabilidade, 
       onde cada elemento representa a probabilidade de pertencer a uma determinada classe.
    """
    resultado2 = sess.run(tf.nn.softmax(resultado1))
    
    """resultado 3: Isso calcula os índices dos valores máximos ao longo do eixo 1 da matriz resultado2. 
       Isso efetivamente retorna a classe prevista para cada exemplo no conjunto de teste.
    """
    resultado3 = sess.run(tf.argmax(resultado2,1))
    
print("Resultado 1:")
print(resultado1)
print("\n")
print("Resultado 2:")
print(resultado2)
print("\n")
print("Resultado 3:")
print(resultado3)

print("\n")

print("y_teste: ")
print(y_teste)
print("\n")
print("y_teste.shape: {}".format(y_teste.shape))


y_teste2 = np.argmax(y_teste, 1)
print("y_teste2: ")
print(y_teste2)

# Marcação de tempo final
end_time = time.time()

# Calcular o tempo decorrido
elapsed_time = end_time - start_time

# Converter segundos em horas, minutos e segundos
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Tempo decorrido: {int(hours)} horas, {int(minutes)} minutos e {seconds:.2f} segundos.")

taxa_acerto = accuracy_score(y_teste2, resultado3)
print("taxa_acerto: {:.2f}%".format(taxa_acerto * 100))
