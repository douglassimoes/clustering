import pandas
import matplotlib.pyplot as pyplot

dataset = pandas.read_csv("datasets/iris.csv",sep=',')
#Linhas por coluna: "
dataset.count()
dataset.head()
#Mais informacoes:")
dataset.describe()
#Flores com tamanho de sepala = 3 e tamanho de petala 1.5 :")
dataset.loc[(dataset["sepal_width"]==3) & (dataset["petal_width"]==1.5)]
#Ordenando os dados pela coluna de comprimento de sepala em ordem descrescente :s")
dataset.sort_values(by="sepal_length", ascending=False)
#Contar a quantidade de flores com sepalas de comprimento 5 :")
dataset[dataset["sepal_length"]==5].count()
#Criando uma coluna categorizada")

def categorize(petal_width):
    if petal_width <= 1.5:
        return None
    elif petal_width <= 3:
        return 'SMA'
    elif petal_width <= 6:
        return 'MED'
    elif petal_width <= 9:
        return 'BIG'

dataset['flower_size'] = dataset["petal_length"].apply(categorize)
dataset[dataset['flower_size']=='BIG']
#Para ver a distribuicao de valores de uma coluna")
pandas.value_counts(dataset['flower_size'])
#Removendo linhas , inplace True subtitui no obj em memoria")
dataset.drop(dataset[dataset["flower_size"]=='SMA'].index,inplace=True)
pandas.value_counts(dataset['flower_size'])
dataset.head()
#Verificar valores faltantes: ")
dataset.isnull().sum()
#Preencher valores nulos com a media da coluna")
dataset['flower_size'].fillna('SMA', inplace=True)
#Remove as linhas com valores nulos: ")
dataset.dropna(inplace=True)
#Plotando graficos")
dataset['species'].value_counts().plot(kind='bar')
dataset.plot(x='species',y='sepal_length', kind='line',color='red')
pyplot.show()
#Removendo uma coluna")
dataset.drop(["flower_size"],axis=1,inplace=True)
