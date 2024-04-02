import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregar o arquivo pré-processado
arquivo_ler = r"C:\Users\GUILHERME\Documents\3_PERÍODO\IC_Daniel\Arquivos_Mao_Robotica\01_hand_close\2022-05-05_14-55-25_filtrado_normalizado.csv"
base = pd.read_csv(arquivo_ler)

# Separar características e rótulosD
X = base.drop('gesture', axis=1)
y = base['gesture']

# Codificar os rótulos numéricos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Criar um modelo K-NN
k = 5  # Número de vizinhos
knn_model = KNeighborsClassifier(n_neighbors=k)

# Treinar o modelo
knn_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = knn_model.predict(X_test)

# Calcular a acurácia do método
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do K-NN: {accuracy:.2f}')
