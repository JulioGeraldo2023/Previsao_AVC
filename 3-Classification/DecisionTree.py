from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    input_files = ['0-Datasets/Avc_Normalization_Min-Max_Avc_Clear_delete.data', 
                   '0-Datasets/Avc_Normalization_Min-Max_Avc_Clear_mean.data', 
                   '0-Datasets/Avc_Normalization_Min-Max_Avc_Clear_median.data', 
                   '0-Datasets/Avc_Normalization_Min-Max_Avc_Clear_mode.data', 
                   '0-Datasets/Avc_Normalization_Min-Max_Avc_Clear_number.data', 
                   
                   '0-Datasets/Avc_Normalization_Robust_Avc_Clear_delete.data',
                   '0-Datasets/Avc_Normalization_Robust_Avc_Clear_mean.data',
                   '0-Datasets/Avc_Normalization_Robust_Avc_Clear_median.data',
                   '0-Datasets/Avc_Normalization_Robust_Avc_Clear_mode.data',
                   '0-Datasets/Avc_Normalization_Robust_Avc_Clear_number.data',
                   
                   '0-Datasets/Avc_Normalization_Unit Vector_Avc_Clear_delete.data',
                   '0-Datasets/Avc_Normalization_Unit Vector_Avc_Clear_mean.data',
                   '0-Datasets/Avc_Normalization_Unit Vector_Avc_Clear_median.data',
                   '0-Datasets/Avc_Normalization_Unit Vector_Avc_Clear_mode.data',
                   '0-Datasets/Avc_Normalization_Unit Vector_Avc_Clear_number.data',
                   
                   '0-Datasets/Avc_Normalization_Z-Score_Avc_Clear_delete.data',
                   '0-Datasets/Avc_Normalization_Z-Score_Avc_Clear_mean.data',
                   '0-Datasets/Avc_Normalization_Z-Score_Avc_Clear_median.data',
                   '0-Datasets/Avc_Normalization_Z-Score_Avc_Clear_mode.data',
                   '0-Datasets/Avc_Normalization_Z-Score_Avc_Clear_number.data']

    for input_file in input_files:
        names = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke'] 
        features = ['age','hypertension','avg_glucose_level','bmi','smoking_status'] 
        target = 'stroke'  
        df = pd.read_csv(input_file, names=names)
        
        # Separating out the features
        X = df.loc[:, features].values
        print(X.shape)

        # Separating out the target
        y = df.loc[:,[target]].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        print(X_train.shape)
        print(X_test.shape)

        clf = DecisionTreeClassifier(max_leaf_nodes=4)
        clf.fit(X_train, y_train)
        tree.plot_tree(clf)
        plt.show()
        
        predictions = clf.predict(X_test)
        print(predictions)
        
        result = clf.score(X_test, y_test)
        print('Acuraccy:')
        print(result)


if __name__ == "__main__":
    main()