import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

X = np.array([[0,0], [0,1], [1,0], [1,1]])

y = np.array([[1], [0], [0], [1]])

neuronios_entrada = 2
neuronios_ocultos = 3
neuronios_saida = 1

#pesos
W = {'oculta': tf.Variable(tf.random_normal([neuronios_entrada, neuronios_ocultos]), name="w_oculta"),
     'saida': tf.Variable(tf.random_normal([neuronios_ocultos, neuronios_saida]), name="w_saida")}

print("type(W): {}".format(type(W)))
print("\n")
print("type(W['oculta']): {}".format(type(W['oculta'])))
print("\n")
print("W['oculta']: {}".format(W['oculta']))
print("\n")
print("W['saida']: {}".format(W['saida']))
print("\n")

# Inicializaçao dos pesos de uma rede neural
distribuicao = np.random.normal(size = 500)
#distribuicao
sns.displot(distribuicao)
#plt.show()

b = {'oculta': tf.Variable(tf.random.normal([neuronios_ocultos]), name='b_oculta'), 
     'saida': tf.Variable(tf.random.normal([neuronios_saida]), name='b_saida')}

xph = tf.placeholder(tf.float32, [4, neuronios_entrada], name='xph') # matriz 4x2
yph = tf.placeholder(tf.float32, [4, neuronios_saida], name = 'yph') # matriz 4x1

camada_oculta = tf.add(tf.matmul(xph, W['oculta']), b['oculta'])
camada_oculta_ativacao = tf.sigmoid(camada_oculta)
camada_saida = tf.add(tf.matmul(camada_oculta_ativacao, W['saida']), b['saida'])
camada_saida_ativacao = tf.sigmoid(camada_saida)
erro = tf.losses.mean_squared_error(yph, camada_saida_ativacao)
otimizador = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(erro)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(W['oculta']))
    print("\n")
    print(sess.run(W['saida']))
    print("\n")
    print("b['oculta']")
    print(sess.run(b['oculta']))
    print("\n")
    print("b['saida']")
    print(sess.run(b['saida']))
    print("\n")
    print('Camada Oculta:')
    print(sess.run(camada_oculta, feed_dict= {xph: X}))
    print("\n")
    print('Camada Oculta Ativacao com a Sigmoide:')
    print(sess.run(camada_oculta_ativacao, feed_dict= {xph: X}))
    print('\n')
    print("Camada Saida: ")
    print(sess.run(camada_saida, feed_dict= {xph: X}))
    print("\n")
    print("Camada Saida Ativacao: ")
    print(sess.run(camada_saida_ativacao, feed_dict= {xph: X}))
    for epocas in range(10000):
        erro_medio = 0
        _, custo = sess.run([otimizador, erro], feed_dict= {xph: X, yph: y})
        if epocas % 200 == 0:
            print(custo)
            erro_medio += custo / 4
            print("Erro Medio:")
            print(erro_medio)
    W_final, b_final = sess.run([W, b])
    
    print("\n")
    print("W_final:")
    print(W_final)
    print("\n")
    print("b_final:")
    print(b_final)
    
    #teste
    camada_oculta_teste = tf.add(tf.matmul(xph, W['oculta']), b['oculta'])
    camada_oculta_ativacao_teste = tf.sigmoid(camada_oculta_teste)
    camada_saida_teste = tf.add(tf.matmul(camada_oculta_ativacao_teste, W_final['saida']), b_final['saida'])
    camada_saida_ativacao_teste = tf.sigmoid(camada_saida_teste)
    
    with tf.Session() as sess:
        sess.run(init)
        print('\n')
        print("Camada Saida Ativacao Teste:")
        print(sess.run(camada_saida_ativacao_teste, feed_dict = {xph: X}))            
    