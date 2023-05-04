import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def ShowInformationDataFrame(df, name):
    print(f"{name}:")
    print(df.head())
    print(df.describe())
    print("\n")
    
def PlotBarChart(df, column_name):
    df[column_name].value_counts().plot(kind='bar')
    plt.title(f'Distribuição de {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Número de Pessoas')
    plt.show()

def main():
    # Faz a leitura do arquivo
    input_file = 'previsao_AVC/0-Datasets/AvcClear1.data'
    names = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']
    features = ['gender','ever_married','work_type','Residence_type','smoking_status']
    df = pd.read_csv(input_file,            # Nome do arquivo com dados
                     names=names,           # Nome das colunas 
                     usecols = features)    # Define as colunas que serão  utilizadas 
                  
    df_original = df.copy()
    ShowInformationDataFrame(df, "Dataframe original")
    
    # Transforma as features categóricas em numéricas
    le = LabelEncoder()
    for feature in features:
        df[feature] = le.fit_transform(df[feature])
        
    ShowInformationDataFrame(df, "Dataframe com features numéricas")
         
    merged_cat = pd.concat([df,df_original],axis=1)
    print(merged_cat)
    
    # Plotar gráficos de barras para todas as colunas categóricas
    for feature in features:
        PlotBarChart(df_original, feature)
    
    # Plotar gráficos de barras para todas as colunas numéricas
    for feature in features:
        if feature not in features and feature != 'stroke':
            PlotBarChart(merged_cat, feature)
        
if __name__ == "__main__":
    main()
    

