# Parte 1: Leitura do csv 'healthcare-datset-strock-data'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

avc = pd.read_csv('../input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv')
avc

# Parte 2: Analisando o dataset

avc = avc.drop('id', axis=1)

print(avc.info())
print()
print('Quantidade de avcs:')
print(avc.stroke.value_counts())

plt.hist(avc.bmi, bins=100)
print('Total de dados \'bmi\':', avc.bmi.count())

# sobrepeso e obesidade sao fatores de risco para desenvolvimento de avc (antigo.saude.gov.br)
cont = 0
cont1 = 0
cont2 = 0
cont3 = 0
for i in range(len(avc.stroke)):
    if avc.bmi[i] > 25:
        cont += 1 
    if avc.bmi[i] > 25 and avc.stroke[i] == 1:
        cont1 += 1 
    if avc.bmi[i] > 25 and avc.avg_glucose_level[i] > 100 and avc.stroke[i] == 1:
        cont2 += 1 
    if avc.bmi[i] > 25 and avc.hypertension[i] == 1 and avc.stroke[i] == 1:
        cont3 += 1
print('Total de pesooas com sobrepeso:',cont)
print('Sofreram avc e tinham sobrepeso: {:.2f}%'.format(cont1/249 * 100))  # 68%   
print('Tinha nivel de glicose alto, sobrepeso e sofreram avc: {:.2f}%'.format(cont2/249 * 100))  # 41%  
print('Tinha hipertensao, sobrepeso e sofreram avc: {:.2f}%'.format(cont3/249 * 100))  # 20% 
# 40 pessoas sem dados 'bmi' sofreram avc ( calculado antes )
print('Caso os 40 dados sem \'bmi\' que sofreram avc estejam com sobrepeso: {:.2f}%'.format((cont1 + 40) / 249 * 100)) # prevendo que os casos sem 'bmi' que sofreram avc também estejam com sobrepeso 

avc['bmi'] = avc['bmi'].fillna(method = 'pad')
plt.hist(avc.bmi, bins=100)
print('Total de dados :', avc.bmi.count())

cont = 0
cont1 = 0
cont2 = 0
cont3 = 0
for i in range(len(avc.stroke)):
    if avc.bmi[i] > 25:
        cont += 1 
    if avc.bmi[i] > 25 and avc.stroke[i] == 1:
        cont1 += 1 
    if avc.bmi[i] > 25 and avc.avg_glucose_level[i] > 100 and avc.stroke[i] == 1:
        cont2 += 1 
    if avc.bmi[i] > 25 and avc.hypertension[i] == 1 and avc.stroke[i] == 1:
        cont3 += 1

print('Total de pesooas com sobrepeso:',cont)
print('Sofreram avc e tinham sobrepeso: {:.2f}%'.format(cont1/249 * 100))  # 82% (diferença de apenas 2% da previsao)   
print('Tinha nivel de glicose alto, sobrepeso e sofreram avc: {:.2f}%'.format(cont2/249 * 100))  # 48%  
print('Tinha hipertensao, sobrepeso e sofreram avc: {:.2f}%'.format(cont3/249 * 100))  # 22% 

print('Work type:')
print(avc.work_type.value_counts())
print()
print('Gender:')
print(avc.gender.value_counts()) # OUTRO ?????????
print()
print('Ever Married:')
print(avc.ever_married.value_counts())   
print()
print('Residence_type')
print(avc.Residence_type.value_counts())

avc['gender'] = avc['gender'].map({'Male':0, 'Female':1 })
avc['ever_married'] = avc['ever_married'].map({'No':0, 'Yes':1})
avc['Residence_type'] = avc['Residence_type'].map({'Urban':0, 'Rural':1})
avc['work_type'] = avc['work_type'].map({'Private':0, 'Self-employed':1, 'children':2, 'Govt_job':3, 'Never_worked':4})
avc

print(avc.smoking_status.value_counts())

avc['smoking_status'] = avc['smoking_status'].map({'never smoked':0, 'Unknown':1, 'formerly smoked':2, 'smokes':3 })

cont = 0 
cont1 = 0
cont2 = 0
for i in range(len(avc.stroke)):
    if avc.age[i] <= 15:
        cont += 1
    if avc.age[i] <= 15 and avc.stroke[i] == 1:
        cont1 += 1
        x = avc.age[i]
    if (avc.smoking_status[i] == 2 or avc.smoking_status[i] == 3 ) and avc.age[i] <= 15:
        cont2 += 1
print('Quantidade total de pessoas com 15 anos ou menos:', cont)
print('Quantidade de pessoas com 15 anos ou menos que sofreram avc:', cont1)
print('Pessoas com 15 anos ou menos que fumam ou ja fumaram:{:.2f}%'.format(cont2/cont * 100)) # 3%
# apenas duas pessoas com 15 anos ou menos sofreu avc e elas não fumava

# todos as pessoas com 15 anos ou menos foram consideradas não fumantes
for i in range(len(avc.stroke)):
    if avc.smoking_status[i] == 1 and avc.age[i] <= 15 :
        avc.smoking_status[i] = 0

avc.smoking_status.value_counts()

cont = 0
cont1 = 0 
cont2 = 0 
for i in range(len(avc.stroke)):
    if avc.smoking_status[i] == 1 and avc.stroke[i] == 1:
        cont += 1
    if (avc.smoking_status[i] == 2 or avc.smoking_status[i] == 3) and avc.stroke[i] == 1:
        cont1 += 1
    if (avc.smoking_status[i] == 2 or avc.smoking_status[i] == 3) and avc.stroke[i] == 1 and ( avc.avg_glucose_level[i] > 100 or avc.hypertension[i] == 1 or avc.heart_disease[i] == 1 ):
        cont2 += 1
        
print('Quantidade de pessoas com status de fumante desconhecido que sofreram avc:', cont)
print('Pessoas que sofreram avc e ja fumaram ou fumam: {:.2f}%'.format(cont1/249 * 100)) # 45%
print('{:.2f}% tinham hipertensao ou alto indice de glicose ou doença cardiaca'.format(cont2/cont1 * 100)) # 76%
print('Das pessoas que tiveram avc e fumavam {:.2f}% NAO tinham hipertensao, alto indice de glicose ou doença cardiaca'.format((cont1 - cont2) /249 * 100)) # 10%

for i in range(len(avc.stroke)):
    if avc.smoking_status[i] == 1 :
        avc.smoking_status[i] = None

avc.smoking_status.replace(np.nan, method = 'pad', inplace=True)
avc.smoking_status.value_counts()

cont1 = 0 
cont2 = 0 
for i in range(len(avc.stroke)):
    if (avc.smoking_status[i] == 2 or avc.smoking_status[i] == 3) and avc.stroke[i] == 1:
        cont1 += 1
    if (avc.smoking_status[i] == 2 or avc.smoking_status[i] == 3) and avc.stroke[i] == 1 and ( avc.avg_glucose_level[i] > 100 or avc.hypertension[i] == 1 or avc.heart_disease[i] == 1 ):
        cont2 += 1
        
        
print('Pessoas que sofreram avc e ja fumaram ou fumam: {:.2f}'.format(cont1/249 * 100)) # 55%
print('{:.2f} tinham hipertensao ou alto indice de glicose ou doença cardiaca'.format(cont2/cont1 * 100)) # 73%
print('Das pessoas que tiveram avc e fumavam {:.2f} NAO tinham hipertensao, alto indice de glicose ou doença cardiaca'.format((cont1 - cont2) /249 * 100)) # 15%

cont = 0 
cont1 = 0
print('Quantidade de dados \'gender\':', avc.gender.count())
for i in range(len(avc.stroke)):
    if avc.gender[i] == 0 and avc.stroke[i] == 1: # male 
        cont += 1
    if avc.gender[i] == 1 and avc.stroke[i] == 1: # female 
        cont1 += 1 
print('quantidade de homens que sofreram avc:', cont)
print('quantidade de mulheres que sofreram avc:', cont1)

cont = 0 
for i in range(len(avc.stroke)):
    if avc.gender[i] != 0 and avc.gender[i] != 1:
        print(avc.loc[i]) # nao sofreu avc
        avc.gender[i] = '0' 

avc.gender.count()

# Parte 3: Aplicando OverSampling



x = avc.drop(['stroke'],axis=1)
y = avc['stroke']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


smote = SMOTE()

x_smote, y_smote = smote.fit_resample(x_train,y_train)
print('Total de strokes do y de treino antes do oversampling de dados: {}'.format(sum(y_train==1)))
print('Total de nao strokes do y de treino antes do oversampling de dados: {} \n'.format(sum(y_train==0)))

print('Shape do x de treino apos o oversampling: {}'.format(x_smote.shape))
print('Shape do y de treino apos o oversampling: {}'.format(y_smote.shape))

print('Total de strokes do y de treino depois do oversampling: {}'.format(sum(y_smote == 1)))
print('Total de nao strokes do y de treino depois do oversampling: {}'.format(sum(y_smote == 0)))

# Parte 4: Aplicando o modelo LogistRegression (antes e depois do oversampling )

model = LogisticRegression()
model.fit(x_train, y_train)

result = model.predict(x_test)

print('Accuracy Score:', accuracy_score(result,y_test))
print('\nConfusion Matrix:')
print(confusion_matrix(result,y_test))
print('\nPrecision:')
print(classification_report(result,y_test))

model.fit(x_smote, y_smote)

result = model.predict(x_test)
print('Accuracy Score:', accuracy_score(result,y_test))
print('\nConfusion Matrix:')
print(confusion_matrix(result,y_test))
print('\nPrecision:')
print(classification_report(result,y_test))