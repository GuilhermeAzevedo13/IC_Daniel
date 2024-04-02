import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

caminho = "C:\\Users\\guilh\\OneDrive\\Documents\\3_periodo_ufs\\IC - Daniel\\Treinamento Phyton\\perceptron\\classificacao_estimators\\classi_estimators\\normalizado_com_gesture_1_e_0.csv"

base = pd.read_csv(caminho)

ch0 = base['ch0']

print(ch0)

comp = len(base)
# print(comp)
fs = 1000
T = 1/fs

t = np.arange(0,comp/fs,T)

print(t)

plt.plot(t, ch0)
plt.show()


f = np.fft.fftfreq(comp,T)
tranfer = np.fft.fft(ch0)
tranfer = np.abs(tranfer)

plt.plot(f[f>0], tranfer[f>0]/comp)

plt.show()