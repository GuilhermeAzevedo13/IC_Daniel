from sklearn import datasets
iris = datasets.load_iris()
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

X = iris.data
print(X.shape)
print(X)
print("\n")
y = iris.target
print(y)
print("\n")

scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
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
print("y.shape:")
print(y.shape)

y = np.array(ct.fit_transform(y), dtype=np.float32)
print("\n")
print(y)


X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.3)

print("X_treinamento.shape:")
print(X_treinamento.shape)

print("X_teste.shape:")
print(X_teste.shape)

neuronios_entrada = X.shape[1]
print("Neuronios_entrada:")
print(neuronios_entrada)

neuronios_oculta  = np.ceil((X.shape[1] + y.shape[1]) / 2)
print("Neuronios_oculta:")
print(neuronios_oculta)

neuronios_saida = y.shape[1]
print("Neuronios_saida:")
print(neuronios_saida)

#pesos
W = {'oculta': tf.Variable(tf.random_normal([int(neuronios_entrada), int(neuronios_oculta)])),
     'saida': tf.Variable(tf.random_normal([int(neuronios_oculta), int(neuronios_saida)]))}


bias = {'oculta': tf.Variable(tf.random_normal([int(neuronios_oculta)])),
        'saida': tf.Variable(tf.random_normal([int(neuronios_saida)]))}

xph = tf.placeholder('float', [None, neuronios_entrada])
yph = tf.placeholder('float', [None, neuronios_saida])

def mlp(x, W, bias):
    camada_oculta = tf.add(tf.matmul(x, W['oculta']), bias['oculta'])
    camada_oculta_ativacao = tf.nn.relu(camada_oculta)
    camada_saida = tf.add(tf.matmul(camada_oculta_ativacao, W['saida']), bias['saida'])
    return camada_saida

modelo = mlp(xph, W, bias)

erro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = modelo, labels = yph))
otimizador = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(erro)

batch_size = 8
batch_total = len(X_treinamento) / batch_size
print("batch_total:")
print(batch_total)
print("\n")

X_batches = np.array_split(X_treinamento, batch_total)
print("X_batches:")
print(X_batches)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoca in range(3000):
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
    resultado1  = sess.run(previsoes_teste, feed_dict= {xph: X_teste})
    resultado2 = sess.run(tf.nn.softmax(resultado1))
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

y_teste2 = np.argmax(y_teste, 1)
print("y_teste2: ")
print(y_teste2)

taxa_acerto = accuracy_score(y_teste2, resultado3)
print("taxa_acerto: {}%".format(taxa_acerto * 100))

