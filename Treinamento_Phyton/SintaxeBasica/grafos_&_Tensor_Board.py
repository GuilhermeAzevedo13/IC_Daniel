import tensorflow as tf

# Pra gerar os grafos no TensorBoard, basta abrir o Anaconda Prompt e inicializar o seguinte
#conda activate CursoTensorFlow36
#tensorboard --logdir=/Users/guilh/output
tf.reset_default_graph()

#a = tf.add(2,2, name = 'add')
#b = tf.multiply(a, 3, name = 'mult1')
#c = tf.multiply(b, a, name = 'mult2')

with tf.name_scope('Operacoes'):
    with tf.name_scope('Escopo_A'):
        a = tf.add(2, 2, name = 'add')
    with tf.name_scope('Escopo_B'):
        b = tf.multiply(a, 3, name = 'mult1')
        c = tf.multiply(b, a, name = 'mult2')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('output', sess.graph)
    print(sess.run(c))
    writer.close()
    
tf.get_default_graph()

grafo1 = tf.get_default_graph()
grafo1

grafo2 = tf.Graph()
grafo2

with grafo2.as_default():
    print(grafo2 is tf.get_default_graph())