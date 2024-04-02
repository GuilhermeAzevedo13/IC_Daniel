# Constantes
import tensorflow as tf

valor1 = tf.constant(2)
valor2 = tf.constant(3)

type(valor1)

print(valor1)

soma = valor1 + valor2

type(soma)

print(soma)

#O tf opera dessa forma criando Tensores:

with tf.Session() as sess:
    s = sess.run(soma)
    
print(s)

#Outro Exemplo: 

texto1 = tf.constant('Texto 1')
texto2 = tf.constant('Texto 2')

type(texto1)

print(texto1)

with tf.Session() as sess:
    con = sess.run(texto1 + texto2)
    
print(con)
