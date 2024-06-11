import pandas as pd

data = pd.read_csv('./penguins.csv')

def convertValues(data):
    # Mapeamento dos valores
    mapeamento_island = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
    mapeamento_sex = {'FEMALE': 0, 'MALE': 1}
    mapeamento_species = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}

    # Aplicando o mapeamento
    data['island'] = data['island'].map(mapeamento_island)
    data['sex'] = data['sex'].map(mapeamento_sex)
    data['species'] = data['species'].map(mapeamento_species)

    return data

data = convertValues(data)

print(data.head())
