import pandas as pd

caminho = "C:\\Users\\guilh\\OneDrive\\Documents\\3_periodo_ufs\\IC - Daniel\\Treinamento Phyton\\perceptron\\normalized150633.csv"

df = pd.read_csv(caminho)

x = df.drop('momento', axis=1)

# Mapeia 'hand_open' para 1 e 'hand_flex_curl' para 0
x['gesture'] = x['gesture'].map({'hand_open': 1, 'hand_flex_curl': 0})

print(x.head())

# Salva o novo DataFrame em um novo arquivo CSV
caminho_novo_arquivo = "C:\\Users\\guilh\\OneDrive\\Documents\\3_periodo_ufs\\IC - Daniel\\Treinamento Phyton\\perceptron\\normalizado150633_com_gesture_1_e_0.csv"
x.to_csv(caminho_novo_arquivo, index=False)
