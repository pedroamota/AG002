import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Leitura do conjunto de dados
data = pd.read_csv('./penguins.csv')

def convertValues(data):
    # Mapa dos valores
    map_island = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
    map_sex = {'FEMALE': 0, 'MALE': 1}
    map_species = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}

    # Aplicando o mapeamento
    data['island'] = data['island'].map(map_island)
    data['sex'] = data['sex'].map(map_sex)
    data['species'] = data['species'].map(map_species)

    return data

def reorderColumns(data):
    new_order = ['island', 'sex', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species']
    data = data.reindex(columns=new_order)

    return data

def trainTestSplit(data):
    # Separação dos dados
    x = data.drop('species', axis=1)
    y = data['species']

    return train_test_split(x, y, test_size=0.2, random_state=42)


# Aplicação de conversões e reordenação
data = convertValues(data)
data = reorderColumns(data)
X_train, X_test, y_train, y_test = trainTestSplit(data)

# Inicialização e treinamento do modelo
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Classificação das amostras de teste e exibição de métricas
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
