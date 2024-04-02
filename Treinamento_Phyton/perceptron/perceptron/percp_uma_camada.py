import tensorflow as tf
import numpy as np

# Dados de entrada
X = np.array([[0.0, 0.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [1.0, 1.0]])

print('Valores de X')
print(X)
print("\n")

# Dados de saída esperados
y = np.array([[0.0], [1.0], [1.0], [1.0]])  # respostas corretas

print("Valores de Y")
print(y)
print("\n")

# Inicialização dos pesos
W = tf.Variable(tf.zeros([2, 1], dtype=tf.float64))  # pesos
def step(x):
    # Função de ativação degrau
    return tf.cast(tf.to_float(tf.math.greater_equal(x, 1)), tf.float64)

print(type(W))

# Inicialização das variáveis TensorFlow
init = tf.global_variables_initializer()

# Cálculo da saída da camada
camada_saida = tf.matmul(X, W)

# Aplicação da função de ativação
camada_saida_ativacao = step(camada_saida)

# Cálculo do erro
erro = tf.subtract(y, camada_saida_ativacao)

# Cálculo do delta e atualização dos pesos
delta = tf.matmul(X, erro, transpose_a=True)
treinamento = tf.assign(W, tf.add(W, tf.multiply(delta, 0.1)))

# Criação de uma sessão TensorFlow
with tf.Session() as sess:
    sess.run(init)
    print("Camada de Saida:")
    print(sess.run(camada_saida))
    
    print("\n")
    print("Camada Saida Ativacao:")
    print(sess.run(camada_saida_ativacao))
    print("\n")
    print("Erro:")
    print(sess.run(erro))
    print("\n")
    print(sess.run(tf.transpose(X)))
    print("\n")
    print("Treinamento:")
    print(sess.run(treinamento))
    print("\n")
    epoca = 0
    # Treinamento por 15 épocas
    for i in range(15):
        epoca += 1
        erro_total, _ = sess.run([erro, treinamento])
        erro_soma = tf.reduce_sum(erro_total)
        print("Época ", epoca, "Erro: ", sess.run(erro_soma))
        # Verifica convergência (erro igual a zero)
        if erro_soma.eval() == 0.0:
            break
    W_final = sess.run(W)
    
print("\n")
print("W_Final")
print(W_final)

# Teste
camada_saida_teste = tf.matmul(X, W_final)
camada_saida_ativacao_teste = step(camada_saida_teste)
with tf.Session() as sess:
    sess.run(init)
    print("\n")
    print("Camada Saida Ativacao Teste: ")
    print(sess.run(camada_saida_ativacao_teste))
