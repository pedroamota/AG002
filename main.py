import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

data = pd.read_csv('./penguins.csv')

def convertValues(data):
    map_island = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
    map_sex = {'FEMALE': 0, 'MALE': 1}
    map_species = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}

    # Aplicando o map
    data['island'] = data['island'].map(map_island)
    data['sex'] = data['sex'].map(map_sex)
    data['species'] = data['species'].map(map_species)

    return data

def reorderColumns(data):
    new_order = ['island', 'sex', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species']
    data = data.reindex(columns=new_order)

    return data

def trainTestSplit(data):
    x = data.drop('species', axis=1)
    y = data['species']

    return train_test_split(x, y, test_size=0.2, random_state=42)

def userInput(model):
    island = int(input("Digite o código da ilha (Biscoe=0, Dream=1, Torgersen=2): "))
    sex = int(input("Digite o sexo do pinguim (Fêmea=0, Macho=1): "))
    culmen_length_mm = float(input("Digite o comprimento do culmen em mm: "))
    culmen_depth_mm = float(input("Digite a profundidade do culmen em mm: "))
    flipper_length_mm = float(input("Digite o comprimento da nadadeira em mm: "))
    body_mass_g = float(input("Digite a massa corporal em gramas: "))
    
    novo_dado = pd.DataFrame({
        'island': [island],
        'sex': [sex],
        'culmen_length_mm': [culmen_length_mm],
        'culmen_depth_mm': [culmen_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g]
    })
    
    especie = model.predict(novo_dado)
    mapeamento_reverso = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
    print(f'A espécie do pinguim é: {mapeamento_reverso[especie[0]]}')

def showTest(X_test, y_pred):
    plt.figure(figsize=(10, 6))
    scatter =  plt.scatter(X_test['culmen_length_mm'], X_test['culmen_depth_mm'], c=y_pred, cmap='viridis', alpha=0.6)

    # Criando a legenda
    classes = ['Adelie', 'Chinstrap', 'Gentoo']
    colors = [scatter.cmap(scatter.norm(value)) for value in range(3)]
    labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]
    plt.legend(labels, classes, title='Espécies Previstas')

    plt.xlabel('Comprimento do Culmen (mm)')
    plt.ylabel('Profundidade do Culmen (mm)')
    plt.show()

data = convertValues(data)
data = reorderColumns(data)
X_train, X_test, y_train, y_test = trainTestSplit(data)

# Treinamento do modelo
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Teste do modelo
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

showTest(X_test, y_pred)
userInput(model)