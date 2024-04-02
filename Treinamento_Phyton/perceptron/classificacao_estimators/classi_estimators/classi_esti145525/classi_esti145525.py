import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

caminho = "C:\\Users\\guilh\\OneDrive\\Documents\\3_periodo_ufs\\IC - Daniel\\Treinamento Phyton\\perceptron\\classificacao_estimators\\classi_estimators\\classi_esti145525\\normalizado145525_com_gesture_1_e_0.csv"

base = pd.read_csv(caminho)
print(base.head())
print("\n")

X = base.drop('gesture', axis = 1)
y = base.gesture

print("X.head")
print(X.head())
print("\n")
print("y.head")
print(y.head())

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.3)

print("X_treinamento.shape: {}".format(X_treinamento.shape))
print("\n")
print("X_teste.shape: {}".format(X_teste.shape))
print("\n")
print("base.columns:")
print(base.columns)
print("\n")

print("Printar a media de cada Canal:")
print("Media do ch0: {}".format(base.ch0.mean()))
print("Media do ch1: {}".format(base.ch1.mean()))
print("Media do ch2: {}".format(base.ch2.mean()))
print("Media do ch3: {}".format(base.ch3.mean()))
print("\n")
print("Printar o Desvio Padrao de cada Canal:")
print("Desvio Padrao de ch0: {}".format(base.ch0.std()))
print("Desvio Padrao de ch1: {}".format(base.ch1.std()))
print("Desvio Padrao de ch2: {}".format(base.ch2.std()))
print("Desvio Padrao de ch3: {}".format(base.ch3.std()))

def padroniza_ch0(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(-0.05852)), tf.constant(0.10586))
def padroniza_ch1(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(-0.04257)), tf.constant(0.17973))
def padroniza_ch2(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(0.08182)), tf.constant(0.14952))
def padroniza_ch3(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(-0.10684)), tf.constant(0.17003))



ch0 = tf.feature_column.numeric_column(key='ch0', normalizer_fn=padroniza_ch0)
ch1 = tf.feature_column.numeric_column(key='ch1', normalizer_fn=padroniza_ch1)
ch2 = tf.feature_column.numeric_column(key='ch2', normalizer_fn=padroniza_ch2)
ch3 = tf.feature_column.numeric_column(key='ch3', normalizer_fn=padroniza_ch3)

colunas = [ch0, ch1, ch2, ch3]

funcao_treinamento = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_treinamento, y = y_treinamento,
                                                        batch_size = 32, num_epochs = None, shuffle = True)

classificador = tf.estimator.DNNClassifier(hidden_units = [256, 256, 256], feature_columns=colunas, n_classes=2)
classificador.train(input_fn = funcao_treinamento, steps = 10000)

funcao_teste = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_teste, y = y_teste, batch_size = 32,
                                                  num_epochs = 1, shuffle = False)

print(classificador.evaluate(input_fn=funcao_teste))

# Definição da função de entrada para predições
funcao_predicao = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_teste, batch_size=32, num_epochs=1, shuffle=False)

# Realizar as previsões no conjunto de teste
previsoes = list(classificador.predict(input_fn=funcao_predicao))
resultados_previsoes = [p['class_ids'][0] for p in previsoes]

# Calcular a acurácia utilizando NumPy
acuracia = np.sum(np.array(resultados_previsoes) == y_teste.values) / len(y_teste)
print("Acurácia no novo conjunto de dados: {:.2f}%".format(acuracia * 100))

###########
###########

# Valores do canal 0
ch0_valores = X_teste['ch0']
# Mapear os acertos e erros no canal 1
acertos_ch0 = ch0_valores[resultados_previsoes == y_teste.values]
erros_ch0 = ch0_valores[resultados_previsoes != y_teste.values]

# Calculando a porcentagem de acertos
total_acertos = len(acertos_ch0)
total_exemplos = len(ch0_valores)
porcentagem_acertos = (total_acertos / total_exemplos) * 100

# Plotar dispersão dos valores do canal 1 e destacar acertos e erros
plt.figure(figsize=(10, 6))

# Plotar valores do canal 1
plt.scatter(range(len(ch0_valores)), ch0_valores, color='blue', label='Valores Ch0')

# Destacar acertos e erros
plt.scatter(np.where(resultados_previsoes == y_teste.values)[0], acertos_ch0, color='green', label='Acertos Ch0')
plt.scatter(np.where(resultados_previsoes != y_teste.values)[0], erros_ch0, color='red', label='Erros Ch0')

plt.xlabel('Exemplos no Conjunto de Teste')
plt.ylabel('Valores do Canal 0')
plt.title(f'Dispersão dos Valores do Canal 0 e Acertos/Erros da Rede Neural\nPorcentagem de Acertos: {porcentagem_acertos:.2f}%')
plt.legend()
plt.grid(True)

# Exibir a porcentagem de acertos no gráfico
plt.text(0.05, 0.95, f'Porcentagem de Acertos: {porcentagem_acertos:.2f}%', transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.show()

###########
###########

# Valores do canal 1
ch1_valores = X_teste['ch1']
# Mapear os acertos e erros no canal 1
acertos_ch1 = ch1_valores[resultados_previsoes == y_teste.values]
erros_ch1 = ch1_valores[resultados_previsoes != y_teste.values]

# Calculando a porcentagem de acertos
total_acertos = len(acertos_ch1)
total_exemplos = len(ch1_valores)
porcentagem_acertos = (total_acertos / total_exemplos) * 100

# Plotar dispersão dos valores do canal 1 e destacar acertos e erros
plt.figure(figsize=(10, 6))

# Plotar valores do canal 1
plt.scatter(range(len(ch1_valores)), ch1_valores, color='blue', label='Valores Ch1')

# Destacar acertos e erros
plt.scatter(np.where(resultados_previsoes == y_teste.values)[0], acertos_ch1, color='green', label='Acertos Ch1')
plt.scatter(np.where(resultados_previsoes != y_teste.values)[0], erros_ch1, color='red', label='Erros Ch1')

plt.xlabel('Exemplos no Conjunto de Teste')
plt.ylabel('Valores do Canal 1')
plt.title(f'Dispersão dos Valores do Canal 1 e Acertos/Erros da Rede Neural\nPorcentagem de Acertos: {porcentagem_acertos:.2f}%')
plt.legend()
plt.grid(True)

# Exibir a porcentagem de acertos no gráfico
plt.text(0.05, 0.95, f'Porcentagem de Acertos: {porcentagem_acertos:.2f}%', transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.show()

