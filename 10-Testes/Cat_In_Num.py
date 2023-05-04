
import pandas as pd

def ShowInformationDataFrame(df, name):
    print(f"{name}:")
    
    # Imprime as 10 primeiras linhas do arquivo
    print("\nPRIMEIRAS 10 LINHAS:\n")
    print(df.head(10))
    
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
    input_file = 'previsao_AVC/0-Datasets/AvcClear1.data'
    output_file = 'previsao_AVC/0-Datasets/Avc_Cat_Num_mode2.data'
    names = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']
    features =  ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']
    df = pd.read_csv(input_file,            # Nome do arquivo com dados
                     names=names,           # Nome das colunas 
                     usecols = features)    # Define as colunas que serão  utilizadas    
    df_original = df.copy() 
    
    #Substitui os valores "F" por 0 e "M" por 1
    df['gender'] = df['gender'].replace({'Male': 0, 'Female': 1, 'Other': 2})
   
    df['ever_married'] = df['ever_married'].replace( {'Yes': 0, 'No': 1})
    
    df['work_type'] = df['work_type'].replace({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4})
    
    df['Residence_type'] = df['Residence_type'].replace({'Urban': 0, 'Rural': 1})
    
    df['smoking_status'] = df['smoking_status'].replace({'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3})
    
    df.to_csv(output_file, header=False, index=False) 

if __name__ == "__main__":
    main()
