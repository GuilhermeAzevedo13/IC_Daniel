from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist = input_data.read_data_sets('mnist/', one_hot = True)

X_treinamento = mnist.train.images
print("X_treinamento.shape: {}".format(X_treinamento.shape))
print("\n")
print("X_treinamento[0]")
print(X_treinamento[0])
print("\n")
y_treinamento = mnist.train.labels
print("y_treinamento.shape: {}".format(y_treinamento.shape))
print("\n")
X_teste = mnist.test.images
print("X_teste.shape: {}".format(X_teste.shape))
print("\n")
y_teste = mnist.test.labels
print("y_teste.shape: {}".format(y_teste.shape))
print("\n")

plt.imshow(X_treinamento[102].reshape((28,28)))
plt.title('Classe: ' + str(np.argmax(y_treinamento[102])))
plt.show()

X_batch, y_batch = mnist.train.next_batch(64)
print("X_batch.shape: {}".format(X_batch.shape))

neuronios_entrada = X_treinamento.shape[1]
print("neuronios_entrada: {}".format(neuronios_entrada))

neuronios_oculta1 = (X_treinamento.shape[1] + y_treinamento.shape[1]) / 2
print("neuronios_oculta1: {}".format(neuronios_oculta1))

neuronios_oculta2 = neuronios_oculta1
neuronios_oculta3 = neuronios_oculta2
neuronios_saida = y_treinamento.shape[1]
print("neuronios_saida {}".format(neuronios_saida))


# 784 -> 397 -> 397 -> 397 -> 10
W = {'oculta1': tf.Variable(tf.random.normal([neuronios_entrada, neuronios_oculta1])),
     'oculta2': tf.Variable(tf.random.normal([neuronios_oculta1, neuronios_oculta2])),
     'oculta3': tf.Variable(tf.random.normal([neuronios_oculta2, neuronios_oculta3])),
     'saida': tf.Variable(tf.random.normal([neuronios_oculta3, neuronios_saida])) 
}

bias = {'oculta1': tf.Variable(tf.random.normal([neuronios_oculta1])),
     'oculta2': tf.Variable(tf.random.normal([neuronios_oculta2])),
     'oculta3': tf.Variable(tf.random.normal([neuronios_oculta3])),
     'saida': tf.Variable(tf.random.normal([neuronios_saida]))   
}

xph = tf.compat.v1.placeholder('float', [None, neuronios_entrada])
yph = tf.compat.v1.placeholder('float', [None, neuronios_saida])

def mlp(x, W, bias):
    camada_oculta1 = tf.nn.relu(tf.add(tf.matmul(x, W['oculta1']), bias['oculta1']))
    camada_oculta2 = tf.nn.relu(tf.add(tf.matmul(camada_oculta1, W['oculta2']), bias['oculta2']))
    camada_oculta3 = tf.nn.relu(tf.add(tf.matmul(camada_oculta2, W['oculta3']), bias['oculta3']))
    camada_saida = tf.add(tf.matmul(camada_oculta3, W['saida']), bias['saida'])
    return camada_saida

modelo = mlp(xph, W, bias)
erro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = modelo, labels = yph))
otimizador = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.0001).minimize(erro)

previsoes = tf.nn.softmax(modelo)
previsoes_corretas = tf.equal(tf.argmax(previsoes, 1), tf.argmax(yph, 1))
taxa_acerto = tf.reduce_mean(tf.cast(previsoes_corretas, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoca in range(5000):
        X_batch, y_batch = mnist.train.next_batch(128)
        _, custo = sess.run([otimizador, erro], feed_dict = {xph: X_batch, yph: y_batch})
        if epoca % 100 == 0:
            acc = sess.run([taxa_acerto], feed_dict = {xph: X_batch, yph: y_batch})
            print('época: ' + str((epoca + 1)) + ' erro: ' + str(custo) + ' acc: ' + str(acc))
            
    print('Treinamento concluído')
    print(sess.run(taxa_acerto, feed_dict = {xph: X_teste, yph: y_teste}))