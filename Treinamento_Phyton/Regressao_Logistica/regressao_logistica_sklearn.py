import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
arquivo_ler = r"C:\Users\guilh\OneDrive\Documents\3º PERIODO UFS\IC - Daniel\Treinamento Phyton\Regressao_Logistica\census.csv"
base = pd.read_csv(arquivo_ler)
print(base.head())

print(base['income'].unique())

#print(base.shape())

#A variavel X vai receber os previsores
x = base.iloc[:, 0:14].values #Pega até o 13

y = base.iloc[:, 14].values

print(x)
print("\n")
print(y)

x[0]

label_encoder = LabelEncoder()
x[:,1] = label_encoder.fit_transform(x[:,1])
x[:,3] = label_encoder.fit_transform(x[:,3])
x[:,5] = label_encoder.fit_transform(x[:,5])
x[:,6] = label_encoder.fit_transform(x[:,6])
x[:,7] = label_encoder.fit_transform(x[:,7])
x[:,8] = label_encoder.fit_transform(x[:,8])
x[:,9] = label_encoder.fit_transform(x[:,9])
x[:,13] = label_encoder.fit_transform(x[:,13])

print("\n")
print(x[0])
print("\n")
print(x[:,1])
print("\n")
print(x[:,3])
print("\n")
print(x[:,5])
print("\n")
print(x[:,6])
print("\n")
print(x[:,7])
print("\n")
print(x[:,8])
print("\n")
print(x[:,9])
print("\n")
print(x[:,13])
print("\n")

scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

print(x[0]) #Dados escalonados

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split()

print(x_treinamento.shape)

print(x_teste.shape)

print