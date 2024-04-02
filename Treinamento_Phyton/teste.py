import tensorflow as tf
import matplotlib 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

x = np.array([[18],[23],[28],[33],[38],[43],[48],[53],[58],[63]])
y = np.array([[871], [1132], [1042], [1356], [1488], [1638], [1569], [1754], [1866], [1900]])

scaler_x = StandardScaler()
X = scaler_x.fit_transform(x)

scaler_y = StandardScaler()
Y = scaler_y.fit_transform(y)
'''
    FORMULA DA REGRESSÃO LINEAR SIMPLES
    y = b0 + b1 * x
'''

np.random.seed(0)

b0 = tf.Variable(0.54)
b1 = tf.Variable(0.71)

erro = tf.losses.mean_squared_error(Y, (b0 + b1 * X)) # Lembrar que o erro ao quadrado penaliza mais os erros maiores.
otimizador = tf.train.GradientDescentOptimizer(learning_rate=0.001)
treinamento = otimizador.minimize(erro) #minimizaçao do erro

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        sess.run(treinamento)
    b0_final = sess.run(b0)
    b1_final = sess.run(b1)

print(b0_final)
print(b1_final)

previsoes = b0_final + b1_final * X

plt.plot(X,Y,'o')
plt.plot(X,previsoes, '*', color = 'red')

previsoes1 = scaler_y.inverse_transform(previsoes)

mae = mean_absolute_error(y, previsoes1)
mse = mean_squared_error(y, previsoes1)

plt.show()

print('erro absoluto: ', mae)
print('erro quadrado: ', mse)