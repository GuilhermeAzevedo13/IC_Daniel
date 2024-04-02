import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("C:\\Users\\guilh\\OneDrive\\Documents\\3_periodo_ufs\\IC - Daniel\\Treinamento Phyton\\testes_filtrados\\normalized.csv")

# Mapeie as classes para valores numéricos
df['gesture'] = df['gesture'].map({'hand_open': 1, 'hand_flex_curl': 0})

# Defina os recursos (X) e o rótulo (y)
X = df[['gesture']]
y = df['ch0']

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crie e treine o modelo KNN
k = 5  # Número de vizinhos
knn = KNeighborsRegressor(n_neighbors=k)
knn.fit(X_train, y_train)

# Realize previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Avalie o desempenho do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Erro Quadrático Médio (MSE): {mse:.2f}')
print(f'Coeficiente de Determinação (R²): {r2:.2f}')



