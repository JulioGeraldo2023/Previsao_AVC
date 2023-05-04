import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    # Faz a leitura de varios arquivo
    input_files = ['previsao_AVC/0-Datasets/Avc_Normalization_Min-Max_Avc_Clear_delete.data', 
                   'previsao_AVC/0-Datasets/Avc_Normalization_Min-Max_Avc_Clear_mean.data', 
                   'previsao_AVC/0-Datasets/Avc_Normalization_Min-Max_Avc_Clear_median.data', 
                   'previsao_AVC/0-Datasets/Avc_Normalization_Min-Max_Avc_Clear_mode.data', 
                   'previsao_AVC/0-Datasets/Avc_Normalization_Min-Max_Avc_Clear_number.data', 
                   
                   'previsao_AVC/0-Datasets/Avc_Normalization_Robust_Avc_Clear_delete.data',
                   'previsao_AVC/0-Datasets/Avc_Normalization_Robust_Avc_Clear_mean.data',
                   'previsao_AVC/0-Datasets/Avc_Normalization_Robust_Avc_Clear_median.data',
                   'previsao_AVC/0-Datasets/Avc_Normalization_Robust_Avc_Clear_mode.data',
                   'previsao_AVC/0-Datasets/Avc_Normalization_Robust_Avc_Clear_number.data',
                   
                   'previsao_AVC/0-Datasets/Avc_Normalization_Unit-Vector_Avc_Clear_delete.data',
                   'previsao_AVC/0-Datasets/Avc_Normalization_Unit-Vector_Avc_Clear_mean.data',
                   'previsao_AVC/0-Datasets/Avc_Normalization_Unit-Vector_Avc_Clear_median.data',
                   'previsao_AVC/0-Datasets/Avc_Normalization_Unit-Vector_Avc_Clear_mode.data',
                   'previsao_AVC/0-Datasets/Avc_Normalization_Unit-Vector_Avc_Clear_number.data',
                   
                   'previsao_AVC/0-Datasets/Avc_Normalization_Z-Score_Avc_Clear_delete.data',
                   'previsao_AVC/0-Datasets/Avc_Normalization_Z-Score_Avc_Clear_mean.data',
                   'previsao_AVC/0-Datasets/Avc_Normalization_Z-Score_Avc_Clear_median.data',
                   'previsao_AVC/0-Datasets/Avc_Normalization_Z-Score_Avc_Clear_mode.data',
                   'previsao_AVC/0-Datasets/Avc_Normalization_Z-Score_Avc_Clear_number.data']
    
    for input_file in input_files:
        names = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke'] 
        features = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status'] 
        target = 'stroke'  
        df = pd.read_csv(input_file, names=names).head(500)    
        
        # restante do código aqui, utilizando o dataframe df como entrada de dados                     
        ShowInformationDataFrame(df,"Dataframe original")

        # Separating out the features
        x = df.loc[:, features].values

        # Separating out the target
        y = df.loc[:,[target]].values

        # PCA projection
        pca = PCA()    
        principalComponents = pca.fit_transform(x)
        print("Explained variance per component:")
        print(pca.explained_variance_ratio_.tolist())
        print("\n\n")

        principalDf = pd.DataFrame(data=principalComponents[:, 0:3],
                           columns=['principal component 1', 'principal component 2', 'principal component 3'])

        finalDf = pd.concat([principalDf, df[[target]]], axis = 1) 
           
        ShowInformationDataFrame(finalDf, "Dataframe PCA")
        
        VisualizePcaProjection(finalDf, target)


def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")
    
           
def VisualizePcaProjection(finalDf, targetColumn):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3 component PCA')

    targets = [0, 1]
    colors = ['r', 'g']

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf[targetColumn] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   finalDf.loc[indicesToKeep, 'principal component 3'],
                   c=color,
                   s=50)

    ax.legend(targets)
    ax.grid()

    plt.show()

if __name__ == "__main__":
    main()





