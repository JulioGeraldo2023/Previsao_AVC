import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
    
def plot_count(df, column_name):
    """
    Plota um gráfico de contagem da coluna column_name no dataframe df.
    """  
    # Verificar valores ausentes no dataframe
    if df.isnull().values.any():
        print("Atenção: existem valores ausentes no dataframe.")

    # Criar gráfico de contagem usando seaborn
    #my_palette = sns.color_palette(["#008B8B", "#FF1493"])
    ax = sns.countplot(data=df, x=column_name, palette="bright")
    #ax = sns.countplot(data=df, x=column_name, palette=my_palette)
    
    # Definir propriedades de fonte
    title_font = fm.FontProperties(weight='bold', size=20)
    label_font = fm.FontProperties(weight='bold', size=14)

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
                height + 35,
                '{:.2f}%'.format(height/total*100),
                ha="center")
        
    # Adicionar grade atrás das colunas
    #plt.grid(True)
    
    # Salvar o gráfico em um arquivo
    #plt.savefig(f"previsao_AVC/9-Graphics/{column_name}_count.png")
    plt.show()
    

def main():
    # Fazer a leitura do arquivo
    input_file = 'previsao_AVC/0-Datasets/Avc_Clear_mode.data'
    names = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke'] 
    features = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status','stroke']
    df = pd.read_csv(input_file, names=names,usecols = features)
    
    # Substituir 0 por "Não" e 1 por "Sim"
    df['heart_disease'] = df['heart_disease'].replace({0: "Doesn't Have Any Heart Diseases", 1: 'Has a Heart Disease'})
    df['stroke'] = df['stroke'].replace({0: "Doesn't Have Stroke", 1: 'Had a Stroke '})
    df['hypertension'] = df['hypertension'].replace({0: "Doesn't Have Hypertension", 1: 'Has Hypertension'})  
    
    # Plotar gráficos de contagem para cada coluna
    for column_name in features:
        plot_count(df, column_name)

if __name__ == "__main__":
    main()


'''

 # Box Plot

    ax = sns.boxplot(data=df,x='stroke',y='age')
    plt.show()
    ax = sns.boxplot(data=df,x='stroke',y='gender')
    plt.show()
    ax = sns.boxplot(data=df,x='stroke',y='hypertension')
    plt.show()
    ax = sns.boxplot(data=df,x='stroke',y='heart_disease')
    plt.show()
    ax = sns.boxplot(data=df,x='stroke',y='ever_married')
    plt.show()
    ax = sns.boxplot(data=df,x='stroke',y='work_type')
    plt.show()
    ax = sns.boxplot(data=df,x='stroke',y='Residence_type')
    plt.show()
    ax = sns.boxplot(data=df,x='stroke',y='avg_glucose_level')
    plt.show()
    ax = sns.boxplot(data=df,x='stroke',y='bmi')
    plt.show()
    ax = sns.boxplot(data=df,x='stroke',y='smoking_status')
    plt.show()
    
    # Scatter plot
    fig, ax = plt.subplots(figsize = (18,10))
    ax.scatter(df['age'], df['stroke'])
    
    # x-axis label
    ax.set_xlabel('(Proporção de Idade)/(AVC)')
    
    # y-axis label
    ax.set_ylabel('Total Age')
    plt.show()
    
    # Lista de valores
    valores = df['age']

    # Calcula o desvio padrão e a média dos valores
    media = np.mean(valores)
    desvio_padrao = np.std(valores)

    # Define o limite para identificar valores discrepantes
    
    #limite = df.loc[df['bmi'] > 40, 'bmi'] = 40
    limite = 3
    # Identifica os valores discrepantes
    outliers = []
    for valor in valores:
        z_score = (valor - media) / desvio_padrao
        if abs(z_score) > limite:
            outliers.append(valor)

    # Substitui os valores discrepantes pela mediana
    mediana = np.median(valores)
    for i, valor in enumerate(valores):
        z_score = (valor - media) / desvio_padrao
        if abs(z_score) > limite:
            valores[i] = mediana

    # Imprime os valores discrepantes
    print("Valores discrepantes substituídos:", outliers)
    
    # Plotando o histograma
    plt.hist(df['age'], bins=20)
    plt.title('Histograma de Valores')
    plt.xlabel('Valores')
    plt.ylabel('Frequência')

    # Destacando os outliers em vermelho
    plt.plot(outliers, [0]*len(outliers), 'ro', markersize=20)

    # Mostrando o gráfico
    plt.show()



    
    #df['age'].unique() # Verificar valores únicos em uma coluna
    #df.loc[df['age'] > 10] # Verificar linhas com valores inconsistentes em uma coluna
    #print(df['age'])
    
    #df.loc[df['bmi'] > 40, 'bmi'] = 40 # Substituir valores inconsistentes por um valor limite
    #df = df.loc[df['bmi'] <= 40] # Remover linhas com valores inconsistentes

    #print(df['bmi'])
    '''