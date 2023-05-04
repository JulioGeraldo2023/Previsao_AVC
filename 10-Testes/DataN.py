from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Lista de valores
valores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000]

# Calcula o desvio padrão e a média dos valores
media = np.mean(valores)
desvio_padrao = np.std(valores)

# Define o limite para identificar valores discrepantes
limite = 3

# Identifica os valores discrepantes
outliers = []
for valor in valores:
    z_score = (valor - media) / desvio_padrao
    if abs(z_score) > limite:
        outliers.append(valor)

# Imprime os valores discrepantes
print("Valores discrepantes:", outliers)



#Min-Max Scaling:
X1 = np.array([[1, 2], [3, 4],[5152, 0.001354]])
scaler = MinMaxScaler()
X_norm1 = scaler.fit_transform(X1)
print("Min-Max")
print(X_norm1)
print("\n")

#Box plot:
X_norm1 = sns.load_dataset('tips')
plt.figure(figsize=(12,8))
sns.boxplot(x=X_norm1['total_bill'])
plt.show()

#Z-score
threshold = 3
outliers5 = np.where(np.abs((X1 - np.mean(X1)) / np.std(X1)) > threshold)[0]
print(outliers5)


##########

#Standardization (z-score normalization):
X2 = np.array([[1, 2], [3, 4],[5152, 0.001354]])
scaler = StandardScaler()
X_norm2 = scaler.fit_transform(X2)
print("Standar")
print(X_norm2)
print("\n")

#Box plot:
X_norm2 = sns.load_dataset('tips')
plt.figure(figsize=(12,8))
sns.boxplot(x=X_norm2['total_bill'])
plt.show()

#Z-score
threshold = 3
outliers = np.where(np.abs((X2 - np.mean(X2)) / np.std(X2)) > threshold)[0]
print(outliers)


##########

#Robust Scaler:
X3 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],[5152, 0.001354]])
scaler = RobustScaler()
X_norm3 = scaler.fit_transform(X3)
print("Robust")
print(X_norm3)
print("\n")

#Box plot:
X_norm3 = sns.load_dataset('tips')
plt.figure(figsize=(12,8))
sns.boxplot(x=X_norm3['total_bill'])
plt.show()

#Z-score
threshold = 3
outliers = np.where(np.abs((X3 - np.mean(X3)) / np.std(X3)) > threshold)[0]
print(outliers)


##########

#Log transformation:
X4 = np.array([[1, 2], [3, 4], [5, 6],[5152, 0.001354]])
X_log = np.log(X4)
print("log")
print(X_log)
print("\n")

#Box plot:
X_log = sns.load_dataset('tips')
plt.figure(figsize=(12,8))
sns.boxplot(x=X_log['total_bill'])
plt.show()

#Z-score
threshold = 3
outliers = np.where(np.abs((X4 - np.mean(X4)) / np.std(X4)) > threshold)[0]
print(outliers)


##########

#Unit Vector Transformation:
X5 = np.array([[1, 2], [3, 4], [5, 6],[5152, 0.001354]])
scaler = Normalizer()
X_norm5 = scaler.fit_transform(X5)
print("Normalizer")
print(X_norm5)
print("\n")

#Box plot:
X_norm5 = sns.load_dataset('tips')
plt.figure(figsize=(12,8))
sns.boxplot(x=X_norm5['total_bill'])
plt.show()

#Z-score
threshold = 3
outliers = np.where(np.abs((X5 - np.mean(X5)) / np.std(X5)) > threshold)[0]
print(outliers)


##########

#data = np.array([1, 2, 3, 4, 5, 10])
q1 = np.percentile(X5, 25)
q3 = np.percentile(X5, 75)
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
outliers1 = np.where((X5 < lower_bound) | (X5 > upper_bound))[0]
print(outliers1)


'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def main():
    # Read the input file
    input_file = 'previsao_AVC/0-Datasets/Avc_Clear_mode.data'
    names = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke'] 
    features = ['age','bmi', 'hypertension','stroke']
    df = pd.read_csv(input_file, names=names, usecols=features)

    # Cria um gráfico 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plota os dados em um gráfico de dispersão 3D
    x = df['age']
    y = df['bmi']
    z = df['hypertension']
    c = df['stroke']
    ax.scatter(x,y,z,c=c)
    #ax.scatter(x, y, z, c=c)

    # Define os rótulos dos eixos
    ax.set_xlabel('Idade')
    ax.set_ylabel('IMC')
    ax.set_zlabel('Nível médio de glicose')

    # Mostra o gráfico
    plt.show()

if __name__ == "__main__":
    main()
    
    


  
  

    
    


 
        
    # Imprima as contagens de valor para as colunas quadro de dados 
    print(df['gender'].value_counts())
    print(df['age'].value_counts())
    print(df['hypertension'].value_counts())
    print(df['heart_disease'].value_counts())
    print(df['ever_married'].value_counts())
    print(df['work_type'].value_counts())
    print(df['Residence_type'].value_counts())
    print(df['avg_glucose_level'].value_counts())
    print(df['bmi'].value_counts())
    print(df['smoking_status'].value_counts())
    print(df['stroke'].value_counts())
    
    print('De %s até %s anos' % (df.gender.min(), df.gender.max()))
    
    print('De %s até %s anos' % (df.hypertension.min(), df.hypertension.max()))
    
    print('De %s até %s anos' % (df.heart_disease.min(), df.heart_disease.max()))
    print('De %s até %s anos' % (df.ever_married.min(), df.ever_married.max()))
    print('De %s até %s anos' % (df.work_type.min(), df.work_type.max()))
    print('De %s até %s anos' % (df.Residence_type.min(), df.Residence_type.max()))
    print('De %s até %s anos' % (df.avg_glucose_level.min(), df.avg_glucose_level.max()))
    print('De %s até %s anos' % (df.bmi.min(), df.bmi.max()))
    print('De %s até %s anos' % (df.smoking_status.min(), df.smoking_status.max()))
    print('De %s até %s anos' % (df.stroke.min(), df.stroke.max()))
    
    # Set the seaborn style to "darkgrid"
    #sns.set_theme(style="darkgrid")

    # Create a countplot using seaborn and the 'gender' column in the 'dataset' dataframe
    #ax = sns.countplot(data=df, x="gender")

    # Show the plot
    #plt.show() 
    
    
    
    
    # Imprima as contagens de valor para as colunas quadro de dados 
    print(df['gender'].value_counts())
    print(df['age'].value_counts())
    print(df['hypertension'].value_counts())
    print(df['heart_disease'].value_counts())
    print(df['ever_married'].value_counts())
    print(df['work_type'].value_counts())
    print(df['Residence_type'].value_counts())
    print(df['avg_glucose_level'].value_counts())
    print(df['bmi'].value_counts())
    print(df['smoking_status'].value_counts())
    print(df['stroke'].value_counts())
    
    # Set the seaborn style to "darkgrid"
    sns.set_theme(style="darkgrid")

    # Create a countplot using seaborn and the 'gender' column in the 'dataset' dataframe
    ax = sns.countplot(data=df, x="gender")

    # Show the plot
    plt.show()
    

    # Gráficos de setores
    fig, ax = plt.subplots(2, 2, figsize=(8, 10))
        
    gender_plot = df['gender'].value_counts()
    hypertension_plot = df['hypertension'].value_counts()
    smoking_status_plot = df['smoking_status'].value_counts()
    stroke_plot = df['stroke'].value_counts()
    
    gender_plot.plot.pie(ax=ax[0,0], autopct='%.2f%%',explode=(0,0,0),colors=['red','blue','orange'])
    hypertension_plot.plot.pie(ax=ax[0,1],labels= 'N''S', autopct='%.2f%%',explode=(0,0.2),colors=['green','red'],)
    smoking_status_plot.plot.pie(ax=ax[1,0], autopct='%.2f%%',explode=(0,0,0,0.2),colors=['green','blue','orange','red'],)
    stroke_plot.plot.pie(ax=ax[1,1],labels= 'N''S', autopct='%.2f%%',explode=(0,0.2),colors=['green','red'],)
    
    ax[0,0].set_title('Gender', fontweight='black')
    ax[0,1].set_title('Hypertension', fontweight='black')
    ax[1,0].set_title('Smoking', fontweight='black')
    ax[1,1].set_title('Stroke', fontweight='black')
    
    # Adiciona um título geral à figura
    fig.suptitle('Graficos em Porcentagem')

    plt.show()
    
    # Set the figure size and create subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    # Plot histogram for age on the first subplot
    df['age'].plot(kind="hist", y="age", bins=70, color="b", ax=axes[0])

    # Plot histogram for bmi on the second subplot
    df['bmi'].plot(kind="hist", y="bmi", bins=100, color="r", ax=axes[1])

    # Plot histogram for avg_glucose_level on the third subplot
    df['avg_glucose_level'].plot(kind="hist", y="avg_glucose_level", bins=100, color="orange", ax=axes[2])

    # Display the plots
    plt.show()
  

#######################

# Faz a leitura do arquivo
    output_file = 'previsao_AVC/0-Datasets/Avc_Clear_num2.data'
    input_file = 'previsao_AVC/0-Datasets/AvcClear.data' 
    names = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke'] 
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas 
     
    df_original = df.copy()
    
    # Imprime as 15 primeiras linhas do arquivo categoricos
    print("PRIMEIRAS 15 LINHAS CATEGORICAS:\n")
    df_cats = df.select_dtypes(include='object')
    print(df_cats.describe())
    print("\n") 
    
    # Imprime as 15 primeiras linhas do arquivo
    #print("PRIMEIRAS 15 LINHAS ONE:\n")
    one_hot_enc = OneHotEncoder(cols=df_cats)
    #print(one_hot_enc)
    print("\n")   
    # Imprime as 15 primeiras linhas do arquivo
    print("PRIMEIRAS 15 LINHAS ONE TRANS:\n")
    df_one_hot_enc = one_hot_enc.fit_transform(df_cats)
    print(df_one_hot_enc)
    print("\n") 
    df_f = pd.concat([df,df_one_hot_enc],axis=1)
    print(df_f)
    df_ff = df_f.drop(df_cats,axis=1,inplace=False)
    print(df_ff)
    
    
    # Salva arquivo com o tratamento para dados faltantes
    df.to_csv(output_file, header=False, index=False) 


'''