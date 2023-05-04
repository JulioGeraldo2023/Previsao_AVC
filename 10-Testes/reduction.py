import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Crie um conjunto de dados com valores aleatórios
input_file = 'previsao_AVC/0-Datasets/Avc_Normalization_Z-Score_Avc_Clear_number.data' 
names = ['id','gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke'] 
features = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']   
df = pd.read_csv(input_file,         # Nome do arquivo com dados
                names = names)      # Nome das colunas 

# Calcule os limites superior e inferior usando o método IQR
q1, q3 = np.percentile(df, [25, 75])
iqr = q3 - q1
lower_limit = q1 - 1.5*iqr
upper_limit = q3 + 1.5*iqr

# Identifique os outliers
outliers = df[(df < lower_limit) | (df > upper_limit)]

# Plote o histograma com os outliers em vermelho
plt.hist(df, bins=50)
plt.axvline(lower_limit, color='r', linestyle='dashed', linewidth=2)
plt.axvline(upper_limit, color='r', linestyle='dashed', linewidth=2)
plt.scatter(outliers, np.zeros_like(outliers), color='r', marker='o')
plt.show()