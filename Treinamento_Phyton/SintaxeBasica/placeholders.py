import tensorflow as tf

#Atribuimos o tipo das variaveis para possibilitar operaçoes sem ter os dados:
#placeholder = Espaço Reservado

p = tf.placeholder('float', None)

operacao = p + 2

with tf.Session() as sess:
    #sess.run(operacao)
    resultado = sess.run(operacao, feed_dict = {p: [1,2,3]})
    print(resultado)
    
p2 = tf.placeholder('float', [None, 5]) # Nao sei quantas linhas tem na minha base de dados, mas sei q tem 5 colunas
operacao2 = p2 * 5

with tf.Session() as sess:
    dados = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    resultadox = sess.run(operacao2, feed_dict={p2: dados})
    print('\n')
    print(resultadox)