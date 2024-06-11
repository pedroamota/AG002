import pandas as pd

data = pd.read_csv('./penguins.csv')

def convertValues(data):
    # Map dos valores
    map_island = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
    map_sex = {'FEMALE': 0, 'MALE': 1}
    map_species = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}

    # Aplicando o map
    data['island'] = data['island'].map(map_island)
    data['sex'] = data['sex'].map(map_sex)
    data['species'] = data['species'].map(map_species)

    return data

data = convertValues(data)

print(data.head())
