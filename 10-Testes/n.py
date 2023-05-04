import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns

def ShowInformationDataFrame(df, name):
    print(f"{name}:")
    print(df.head())
    print(df.describe())
    print("")

def main():
    # Faz a leitura do arquivo
    input_file = 'previsao_AVC/0-Datasets/Avc_Cat_Num_mode.data'
    output_file = 'previsao_AVC/0-Datasets/Avc_Normalization_Min-Max_mode.data'
    names = ['age','hypertension','heart_disease','avg_glucose_level','bmi','stroke','gender_num','ever_married_num','work_type_num','Residence_type_num','smoking_status_num']
    features = ['age','hypertension','heart_disease','avg_glucose_level','bmi','gender_num','ever_married_num','work_type_num','Residence_type_num','smoking_status_num']
    target = 'stroke'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names=names) # Nome das colunas    
             
    ShowInformationDataFrame(df, "Dataframe numérico")

    
    #Box plot:
    #df = sns.load_dataset('tips')
    plt.figure(figsize=(12,8))
    ax = sns.boxplot(data=df)
    plt.show()
    
    ojt_pad = StandardScaler().fit(df)
    df_pad = ojt_pad.transform(df)
    print(df_pad)
    
    df_pad = pd.DataFrame(df_pad)
    df_pad.columns = ['age','hypertension','heart_disease','avg_glucose_level','bmi','stroke','gender_num','ever_married_num','work_type_num','Residence_type_num','smoking_status_num']

    print(df_pad.describe())
    
    plt.figure(figsize=(12,8))
    ax = sns.boxplot(data=df_pad)
    plt.show()
    
    ojt_nor = MinMaxScaler().fit(df)
    df_nor = ojt_nor.transform(df)
    print(df_nor)
    
    df_nor = pd.DataFrame(df_nor)
    df_nor.columns = ['age','hypertension','heart_disease','avg_glucose_level','bmi','stroke','gender_num','ever_married_num','work_type_num','Residence_type_num','smoking_status_num']

    print(df_nor.describe())
    
    plt.figure(figsize=(12,8))
    ax = sns.boxplot(data=df_nor)
    plt.show()
    
    '''
    # Separating out the features
    x = df.loc[:, features].values
   
    # Separating out the target
    y = df.loc[:, [target]].values

    # Z-score normalization
    x_zscore = StandardScaler().fit_transform(x)
    normalized1_df = pd.DataFrame(data=x_zscore, columns=features)
    normalized1_df = pd.concat([normalized1_df, df[[target]]], axis=1)
    ShowInformationDataFrame(normalized1_df, "Dataframe Z-Score Normalized")

    # Min-Max normalization
    x_minmax = MinMaxScaler().fit_transform(x)
    normalized2_df = pd.DataFrame(data=x_minmax, columns=features)
    normalized2_df = pd.concat([normalized2_df, df[[target]]], axis=1)
    ShowInformationDataFrame(normalized2_df, "Dataframe Min-Max Normalized")

    #normalized1_df.to_csv(output_file, header=False, index=False) 
    #normalized2_df.to_csv(output_file, header=False, index=False) 
    '''
if __name__ == "__main__":
    main()
    

