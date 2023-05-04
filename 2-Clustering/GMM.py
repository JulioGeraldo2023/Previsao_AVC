'''
Este código implementa o algoritmo K-means a partir do zero e 
também usa a biblioteca Scikit-learn para treinar um modelo de 
Mistura Gaussiana (Gaussian Mixture Model - GMM) e visualizar 
os resultados em um conjunto de dados de imagens de dígitos 
manuscritos (o conjunto de dados MNIST).

Primeiro, o conjunto de dados é carregado usando a função 
load_digits() da biblioteca Scikit-learn. Em seguida, as 
imagens são exibidas usando a função show_digitsdataset().

Em seguida, o conjunto de dados é reduzido para 2 dimensões 
usando PCA (Análise de Componentes Principais) para que possa 
ser visualizado em um gráfico. Os dois componentes principais 
(ou eixos) que explicam a maior variação nos dados são plotados 
usando a função plot_samples().

Depois disso, a biblioteca Scikit-learn é usada para ajustar 
um modelo GMM com 10 componentes usando os dados projetados. 
As proporções de peso e médias dos componentes são impressas 
na saída. O modelo treinado é usado para prever os rótulos 
de classe para cada imagem usando a função predict().

Por fim, os resultados são plotados novamente usando a função 
plot_samples(), com os rótulos de classe previstos em vez dos 
rótulos originais. Os gráficos resultantes mostram como as 
imagens de dígitos manuscritos são agrupadas em clusters pelos 
algoritmos K-means e GMM.
'''

#Implementation of Kmeans from scratch and using sklearn
#Loading the required modules 
import numpy as np
from scipy.spatial.distance import cdist 
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import pandas as pd

def plot_samples(projected, labels, title):    
    fig = plt.figure()
    u_labels = np.unique(labels)
    #for i in u_labels:
    plt.scatter(projected[labels , 0] , projected[labels , 1] ,
                    edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('tab10', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend()
    plt.title(title)

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
        #ShowInformationDataFrame(df,"Dataframe original")

        # Separating out the features
        x = df.loc[:, features].values

        # Separating out the target
        y = df.loc[:,[target]].values

        # PCA projection
        pca = PCA(2)    
        projected = pca.fit_transform(x)
        print("Explained variance per component:")
        print(pca.explained_variance_ratio_.tolist())
        print("\n\n")
        print(x.shape)
        print(projected.shape)    
        plot_samples(projected, y, 'Original Labels') 
    
        #Applying sklearn GMM function
        gm  = GaussianMixture(n_components=10).fit(projected)
        print(gm.weights_)
        print(gm.means_)
        x = gm.predict(projected)

        #Visualize the results sklearn
        plot_samples(projected, x, 'Clusters Labels GMM')

        plt.show()

if __name__ == "__main__":
    main()