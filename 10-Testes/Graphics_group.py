import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
        
def create_bins(df, column_name, bins, labels):
    df[column_name] = pd.cut(df[column_name], bins=bins, labels=labels)

def plot_groups(df, column_name):
    """
    Plota um gráfico de contagem das faixas de idade no dataframe df.
    """  
    # Verificar valores ausentes no dataframe
    if df.isnull().values.any():
        print("Atenção: existem valores ausentes no dataframe.")

    # Criar faixas de idade
    if column_name == "age":
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90"]
    # Criar faixas de indice de massa corporal
    elif column_name == "bmi":
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
    # Criar faixas de avg_glucose_level
    elif column_name == "avg_glucose_level":
        bins = [0, 50, 100, 150, 200, 250, 300]
        labels = ["0-50", "50-100", "100-150", "150-200", "200-250", "250-300"]

    create_bins(df, column_name, bins, labels)
    
    # Criar gráfico de contagem usando seaborn
    ax = sns.countplot(data=df, x=column_name, palette="bright")
        
    # Definir propriedades de fonte
    title_font = fm.FontProperties(weight='bold', size=20)
    label_font = fm.FontProperties(weight='extra bold', size=14)

    # Adicionar título e rótulos aos eixos com as propriedades de fonte definidas
    # Adicionar título e rótulos aos eixos
    ax.set_title(f"Count of {column_name}",fontproperties=title_font)
    ax.set_xlabel(column_name, fontproperties=label_font)
    ax.set_ylabel("Numbers Of People", fontproperties=label_font)

    # Adicionar porcentagens aos rótulos do eixo Y
    total = float(len(df))
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:.2f}%'.format(height/total*100),
                ha="center")

    # Salvar o gráfico em um arquivo
    #plt.savefig(f"previsao_AVC/9-Graphics/{column_name}_count.png")
    plt.show()
    
def main():
    # Fazer a leitura do arquivo
    input_file = 'previsao_AVC/0-Datasets/Avc_Clear_mode.data'
    names = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke'] 
    features = ['age','avg_glucose_level','bmi']
    df = pd.read_csv(input_file, names=names,usecols = features)

    # Plotar gráficos de contagem para cada coluna
    for column_name in features:
        plot_groups(df, column_name)

if __name__ == "__main__":
    main()
