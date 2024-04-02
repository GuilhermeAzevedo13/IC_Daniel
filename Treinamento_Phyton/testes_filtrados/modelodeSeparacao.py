import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar o conjunto de dados do arquivo CSV
df = pd.read_csv("C:\\Users\\guilh\\OneDrive\\Documents\\3_periodo_ufs\\IC - Daniel\\Treinamento Phyton\\testes_filtrados\\normalized.csv")

# Dividindo o conjunto de dados em treinamento (80%) e teste (20%)
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Salvando os conjuntos de treinamento e teste em arquivos CSV separados
train_data.to_csv("treinamento.csv", index=False)
test_data.to_csv("teste.csv", index=False)
