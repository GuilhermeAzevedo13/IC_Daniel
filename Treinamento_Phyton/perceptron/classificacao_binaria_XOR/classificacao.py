import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.metrics import accuracy_score

caminho_do_arquivo_original = "C:\\Users\\guilh\\OneDrive\\Documents\\3_periodo_ufs\\IC - Daniel\\Treinamento Phyton\\perceptron\\classificacao_binaria_XOR\\normalizado_com_gesture_1_e_0.csv"

df = pd.read_csv(caminho_do_arquivo_original)

data_array = np.array(df)

# Separando as colunas
X = data_array[:, 2:6]  # Colunas 'ch0', 'ch1', 'ch2', 'ch3'
y = data_array[:, 1].reshape(-1, 1)  # Coluna 'gesture'

# print(X.shape)
# print(y)
# print(y.shape)

neuronios_entrada = 4
neuronios_ocultos = 50
neuronios_saida = 1

# pesos - sao 24 pesos
W = {'oculta': tf.Variable(tf.random_normal([neuronios_entrada, neuronios_ocultos]), name="w_oculta"),
     'saida': tf.Variable(tf.random_normal([neuronios_ocultos, neuronios_saida]), name="w_saida")}

# print("type(W): {}".format(type(W)))
# print("\n")
# print("type(W['oculta']): {}".format(type(W['oculta'])))
# print("\n")
# print("W['oculta']: {}".format(W['oculta']))
# print("\n")
# print("W['saida']: {}".format(W['saida']))
print("\n")

# Inicializaçao dos pesos de uma rede neural
distribuicao = np.random.normal(size=500)  # vamos gerar 500 numeros aleatorios
# distribuicao
sns.displot(distribuicao)
# plt.show()

bias = {'oculta': tf.Variable(tf.random.normal([neuronios_ocultos]), name='b_oculta'),
        'saida': tf.Variable(tf.random.normal([neuronios_saida]), name='b_saida')}

xph = tf.placeholder(
    tf.float32, [119997, neuronios_entrada], name='xph')  # matriz 119997x4
yph = tf.placeholder(
    tf.float32, [119997, neuronios_saida], name='yph')  # matriz 119997x1

camada_oculta = tf.add(tf.matmul(xph, W['oculta']), bias['oculta'])
camada_oculta_ativacao = tf.sigmoid(camada_oculta)
camada_saida = tf.add(
    tf.matmul(camada_oculta_ativacao, W['saida']), bias['saida'])
camada_saida_ativacao = tf.sigmoid(camada_saida)

# vai encontrar o menor erro possivel entre os valores reais: yph e os valores encontrados: camada_saida_ativacao
erro = tf.losses.mean_squared_error(yph, camada_saida_ativacao)
otimizador = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(
    erro)  # mexer no learning rate para esse banco de dados

init = tf.global_variables_initializer()

# Marcação de tempo inicial
start_time = time.time()

with tf.Session() as sess:
    sess.run(init)
    # print(sess.run(W['oculta']))
    # print("\n")
    # print(sess.run(W['saida']))
    # print("\n")
    # print("bias['oculta']")
    # print(sess.run(bias['oculta']))
    # print("\n")
    # print("bias['saida']")
    # print(sess.run(bias['saida']))
    # print("\n")
    # print('Camada Oculta:')
    # print(sess.run(camada_oculta, feed_dict= {xph: X}))
    # print("\n")
    # print('Camada Oculta Ativacao com a Sigmoide:')
    # print(sess.run(camada_oculta_ativacao, feed_dict= {xph: X}))
    # print('\n')
    # print("Camada Saida: ")
    # print(sess.run(camada_saida, feed_dict= {xph: X}))
    print("\n")
    print("Camada Saida Ativacao com a Sigmoide: ")
    print(sess.run(camada_saida_ativacao, feed_dict={xph: X}))
    print("\n")

    max_epocas = 50000
    for epocas in range(max_epocas):
        erro_medio = 0
        _, custo = sess.run([otimizador, erro], feed_dict={xph: X, yph: y})
        if epocas % 200 == 0:
            # print(custo)
            erro_medio += (custo / 4)
            print("Erro Medio:")
            print(erro_medio)
    W_final, bias_final = sess.run([W, bias])

    print("\n")
    print("W_final:")
    print(W_final)
    print("\n")
    print("bias_final:")
    print(bias_final)

    # teste
    camada_oculta_teste = tf.add(tf.matmul(xph, W['oculta']), bias['oculta'])
    camada_oculta_ativacao_teste = tf.sigmoid(camada_oculta_teste)
    camada_saida_teste = tf.add(
        tf.matmul(camada_oculta_ativacao_teste, W_final['saida']), bias_final['saida'])
    camada_saida_ativacao_teste = tf.sigmoid(camada_saida_teste)

    # Threshold para converter probabilidades em classes (0 ou 1)
    threshold = 0.5

    with tf.Session() as sess:
        sess.run(init)
        print('\n')
        camada_saida_ativacao_teste = sess.run(
            camada_saida_ativacao_teste, feed_dict={xph: X})
        print("Camada Saida Ativacao Teste:")
        print(camada_saida_ativacao_teste)

    # Converter as saídas para classes binárias usando o threshold
    previsoes = (camada_saida_ativacao_teste > threshold).astype(int)

    # Converter os rótulos reais para classes binárias
    y_binario = (y > threshold).astype(int)

    # Calcular a acurácia
    acuracia = np.mean(previsoes == y_binario)
    print("Acurácia do treinamento:", acuracia)

    # Marcação de tempo final
    end_time = time.time()

    # Calcular o tempo decorrido
    elapsed_time = end_time - start_time

    # Converter segundos em horas, minutos e segundos
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(
        f"Tempo decorrido: {int(hours)} horas, {int(minutes)} minutos e {seconds:.2f} segundos.")
    
    taxa_acerto = accuracy_score(y, y_binario)
    print("taxa_acerto: {:.2f}%".format(taxa_acerto * 100))

    # Plotagem
    plt.plot(camada_saida_ativacao_teste, label='Camada Saida Ativacao Teste')
    plt.xlabel('Exemplos')
    plt.ylabel('Valores')
    plt.title('Comportamento da Camada Saida Ativacao Teste')
    plt.legend()
    plt.show()
