


import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import seaborn as sns

def main():
    # Faz a leitura do arquivo
    names = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke'] 
    features = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']   
    input_file = 'previsao_AVC/0-Datasets/AvcClear1.data' 
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names = names,      # Nome das colunas 
                     usecols = features)


    # check correlation
    # Calculate correlation matrix and visualize as heatmap
    plt.figure(figsize=(14,6))
    dfc=abs(df.corr())
    sns.heatmap(dfc,annot=True,linewidth=1)
    plt.show()
    # Plot feature correlations with target variable
    plt.figure(figsize=(10,6))
    plt.plot(dfc["stroke"].sort_values(ascending=False)[1:],label="Correlation")
    plt.ylabel("Correlation")
    plt.xlabel("Feature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    '''  
    #### 1º Métodos para transformar categoricos em numericos ####
    
    # Cria uma instância do LabelEncoder
    le = LabelEncoder()

    # Codifica a coluna 'genero'
    df['gender'] = le.fit_transform(df['gender'])
    df['ever_married'] = le.fit_transform(df['ever_married'])
    df['work_type'] = le.fit_transform(df['work_type'])
    df['Residence_type'] = le.fit_transform(df['Residence_type'])
    df['smoking_status'] = le.fit_transform(df['smoking_status'])
  

    # Usa o método loc para atribuir os novos valores numéricos na coluna original
    df.loc['Male':, 'gender'] = 0
    df.loc['Yes': , 'ever_married'] = 0
    df.loc['Private':, 'work_type'] = 0
    df.loc['Urban': , 'Residence_type'] = 0
    df.loc['formerly smoked':, 'smoking_status'] = 0
    print(df)
    
    #### 2º Métodos para transformar categoricos em numericos ####
    
    # Converte os atributos categóricos para numéricos
        # get all categorical columns
    cat_columns = df.select_dtypes(['object']).columns

        # convert all categorical columns to numeric
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    
    
    
    #### 3º Métodos para transformar categoricos em numericos ####
    
    df_numerical = df_line
    
    #Substitui os valores Categóricos por Numéricos
    df_numerical['gender'] = df_line['gender'].replace({'Male': 0, 'Female': 1, 'Other': 2})
   
    df_numerical['ever_married'] = df_line['ever_married'].replace( {'Yes': 0, 'No': 1})
    
    df_numerical['work_type'] = df_line['work_type'].replace({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4})
    
    df_numerical['Residence_type'] = df_line['Residence_type'].replace({'Urban': 0, 'Rural': 1})
    
    df_numerical['smoking_status'] = df_line['smoking_status'].replace({'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}) 
          
    ShowInformationDataFrame(df_numerical, "\nDataframe Dados Numéricos")
    '''
if __name__ == "__main__":
    main()
    
   
 