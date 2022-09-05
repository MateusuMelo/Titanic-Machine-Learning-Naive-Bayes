import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import joblib
from sklearn.metrics import confusion_matrix



#importando dados
data_set = pd.read_csv("train.csv")
#filtrando dados
data_set.drop(['Fare', 'Ticket','SibSp','Parch','Name','Cabin'], inplace=True,axis=1) #Removendo colunas 'inuteis' para as previsões
data_set['Sex'] = data_set['Sex'].map({'male': 1,'female':0}) #Mapeando coluna Sex p binario
data_set['Embarked'] = data_set['Embarked'].map({'S':0,'Q':1,'C':2})
data_treino, data_teste = sk.model_selection.train_test_split(data_set,test_size=0.2,random_state=42) #dividindo dataset

#Removendo valores faltantes
data_treino = data_treino.fillna(200)
data_teste = data_teste.fillna(200)

X_treino = data_treino.drop(['Survived','PassengerId'],axis=1)
y_treino = data_treino['Survived']
X_teste = data_teste.drop(['Survived','PassengerId'],axis=1)
y_teste = data_teste['Survived']

nb = BernoulliNB()
y_prev = nb.fit(X_treino,y_treino)
X_final = X_teste.assign(Survived=y_prev.predict(X_teste)) # Gerando um dataframe acoplando as previsoes  na coluna survived
#salvando modelo
joblib.dump(y_prev,'Titanic_Naive_Bayes.pkl')

#Previsões do arquivo de testes do keagle
testes = pd.read_csv("test.csv")
modelo = joblib.load('Titanic_Naive_Bayes.pkl')

#filtrando testes para aplicar previsão
testes_flt = testes
testes_flt = testes_flt.fillna(200)
testes_flt.drop(['Fare', 'Ticket','SibSp','Parch','Name','Cabin','PassengerId'],inplace=True,axis=1)
testes_flt['Sex'] = testes_flt['Sex'].map({'male': 1,'female':0})
testes_flt['Embarked'] = testes_flt['Embarked'].map({'S':0,'Q':1,'C':2})

#Realizando previsão e criando dataframe ja previsto
previsao = modelo.predict(testes_flt)
data_final = pd.read_csv("test.csv")
testes_final = data_final.assign(Survived = previsao)
testes_final[['PassengerId','Survived']].to_csv('test_prev.csv',index=False)

#Resultado
gabarito = pd.read_csv('gender_submission.csv')
print(confusion_matrix(gabarito['Survived'],testes_final['Survived']))


