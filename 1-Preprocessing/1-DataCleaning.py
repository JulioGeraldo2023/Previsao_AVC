import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

'''
Limpeza de dados é o processo de identificar e corrigir erros, 
inconsistências e dados incompletos ou duplicados em um conjunto de dados. 
Esse processo é essencial para garantir que os dados sejam precisos, 
animadores e úteis para análise e tomada de decisão.

Durante o processo de limpeza de dados, podem ser realizadas 
diversas atividades, como remover valores duplicados, 
corrigir valores inválidos ou inconsistentes, preencher dados faltantes, 
padronizar formatos de dados e remover dados irrelevantes ou inconsistentes. 
O objetivo é garantir que os dados estejam em um estado em que possam 
ser analisados com segurança e precisão.
'''

def ShowInformationDataFrame(df, name):
    print(f"{name}:")
    
    # Imprime as 10 primeiras linhas do arquivo
    print("\nPRIMEIRAS 10 LINHAS:\n")
    print(df.head(10).T)
    
    # Imprime informações sobre dos dados
    print("\nINFORMAÇÕES GERAIS DOS DADOS\n")
    print(df.info())
    
    # Imprime uma analise descritiva sobre dos dados
    print("\nDESCRIÇÃO DOS DADOS\n")
    print(df.describe())
    
    # Imprime a quantidade de valores faltantes por coluna
    print("\nVALORES FALTANTES")
    missing_values = df.isnull().sum()
    print(f'\n{missing_values}\n')
    
    # Calcular quantidade e porcentagem de valores faltantes por coluna
    for col in df.columns:
        num_missing = missing_values[col]
        pct_missing = num_missing / len(df) * 100
        print(f'{num_missing} valores faltantes correspondem a ({pct_missing:.1f}%) da coluna {col}')

def main():
    # Faz a leitura do arquivo
    input_file = 'previsao_AVC/0-Datasets/Avc.csv' 
    names = ['id','gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke'] 
    features = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']   
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names = names,      # Nome das colunas 
                     usecols = features, # Define as colunas que serão  utilizadas
                     na_values='N/A')    # Define que N/A será considerado valores ausentes
     
    df_original = df.copy()
    
    ShowInformationDataFrame(df, "\nDataframe Original")  
      
    # Verificar linhas duplicadas
    duplicated_df = df[df.duplicated(keep=False)]
    if not duplicated_df.empty:
        print("\nLinhas duplicadas:\n")
        print(duplicated_df)  
        
    # Apagar linhas duplicadas  
    df_line = df.drop_duplicates(keep='first')
    print("\nLinha apagada:\n")
    print(df[~df.isin(df_line)].dropna())
    
    ShowInformationDataFrame(df_line, "\nDataframe Linha Duplicadas Apagadas") 
    
    #### Métodos para transformar categoricos em numericos ####
    
    df_numerical = df_line
    
    # Substitui os valores Categóricos por Numéricos
    df_numerical['gender'] = df_line['gender'].replace({'Male': 0, 'Female': 1, 'Other': 2})
   
    df_numerical['ever_married'] = df_line['ever_married'].replace( {'Yes': 0, 'No': 1})
    
    df_numerical['work_type'] = df_line['work_type'].replace({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4})
    
    df_numerical['Residence_type'] = df_line['Residence_type'].replace({'Urban': 0, 'Rural': 1})
    
    df_numerical['smoking_status'] = df_line['smoking_status'].replace({'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}) 
          
    ShowInformationDataFrame(df_numerical, "\nDataframe Dados Numéricos")
            
    columns_missing_value = df_numerical.columns[df_numerical.isnull().any()]
    print(columns_missing_value)
    method = None # number or median or mean or mode or delete
    
    for c in columns_missing_value:
        UpdateMissingValues(df_numerical, c, method)
  
    # Salva arquivo com o tratamento para dados faltantes para cada método
    for method in ['number', 'median', 'mean', 'mode', 'delete']:
        output_file = f'previsao_AVC/0-Datasets/Avc_Clear_{method}.data'
        df_method = df_numerical.copy()
        for c in columns_missing_value:
            UpdateMissingValues(df_method, c, method)
        #df_method.to_csv(output_file, header=False, index=False) # Indice não será salvo no arquivo (index=False)
        
    ShowInformationDataFrame(df_method, "\nDataframe Com Dados Numéricos Sem Valores Ausentes")

def UpdateMissingValues(df_numerical, column, method=None, number=None):
    if method == 'number':
        # Substituindo valores ausentes por um número
        number = df_numerical[column].quantile(q=0.75) # será igual a 75% do total 
        df_numerical[column].fillna(number, inplace=True)
    elif method == 'median':
        # Substituindo valores ausentes pela mediana 
        median = df_numerical[column].median()
        df_numerical[column].fillna(median, inplace=True)
    elif method == 'mean':
        # Substituindo valores ausentes pela média
        mean = round(df_numerical[column].mean(), 1)   
        df_numerical[column].fillna(mean, inplace=True)
    elif method == 'mode':
        # Substituindo valores ausentes pela moda
        mode = df_numerical[column].mode()[0]
        df_numerical[column].fillna(mode, inplace=True)
    elif method == 'delete':
        # Excluindo valores ausentes 
        df_numerical.dropna(subset=[column], inplace=True)

if __name__ == "__main__":
    main()
 











