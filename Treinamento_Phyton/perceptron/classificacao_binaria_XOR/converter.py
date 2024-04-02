import pandas as pd

caminho_do_arquivo_original = "C:\\Users\\guilh\\OneDrive\\Documents\\3_periodo_ufs\\IC - Daniel\\Treinamento Phyton\\perceptron\\classificacao_binaria_XOR\\normalized.csv"

df = pd.read_csv(caminho_do_arquivo_original)

# Mapeia 'hand_open' para 1 e 'hand_flex_curl' para 0
df['gesture'] = df['gesture'].map({'hand_open': 1, 'hand_flex_curl': 0})

# Salva o novo DataFrame em um novo arquivo CSV
caminho_novo_arquivo = "C:\\Users\\guilh\\OneDrive\\Documents\\3_periodo_ufs\\IC - Daniel\\Treinamento Phyton\\perceptron\\classificacao_binaria_XOR\\normalizado_com_gesture_1_e_0.csv"
df.to_csv(caminho_novo_arquivo, index=False)


