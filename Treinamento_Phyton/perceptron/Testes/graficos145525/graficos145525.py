import pandas as pd
import matplotlib.pyplot as plt

caminho = "C:\\Users\\guilh\\OneDrive\\Documents\\3_periodo_ufs\\IC - Daniel\\Treinamento Phyton\\perceptron\\Testes\\graficos145525\\normalizado145525_com_gesture_1_e_0.csv"

df = pd.read_csv(caminho)

print(df.head())

x = range(len(df))
y = df['ch0']

plt.plot(x, y)
plt.xlabel('Quantidade de Dados do Channel 0')
plt.ylabel('Normalização')
plt.title('Sinal Miográfico - Channel 0')

# Definir limites para o eixo y entre -1 e 1
plt.ylim(-1, 1)
plt.grid(True)
plt.show()

contagem_gestures = df['gesture'].value_counts()

# Exibindo a contagem
print(contagem_gestures)
