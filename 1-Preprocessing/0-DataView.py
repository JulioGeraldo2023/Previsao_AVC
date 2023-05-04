import pandas as pd     # Biblioteca para manipulação de dados em formato tabular
import numpy as np      # Biblioteca para cálculos numéricos
import matplotlib.pyplot as plt    # Biblioteca para criação de gráficos
import seaborn as sns   # Biblioteca para visualização de dados
import matplotlib.font_manager as fm

def create_bins(df, column_name, bins, labels):
    df[column_name + '_bins'] = pd.cut(df[column_name], bins=bins, labels=labels)
    return df

def PlotBarChart(df, column_name_num):
    """
    Plota um gráfico de contagem das faixas de idade no dataframe df.
    """  
    # Verificar valores ausentes no dataframe
    if df.isnull().values.any():
        print("Atenção: existem valores ausentes no dataframe.")
        
    '''
    A densidade dos dados se refere à quantidade de informações contidas em uma determinada área ou intervalo. 
    No contexto de análise de dados, a densidade pode ser aberta como a razão entre o número de observações 
    em uma determinada faixa e a largura dessa faixa. Isso pode ser útil para entender como os dados 
    estão distribuídos e se há concentrações ou dispersões em determinadas regiões. 
    A densidade também pode ser visualizada graficamente por meio de histogramas ou gráficos de densidade, 
    que mostram como observações se distribuem ao longo de um eixo.
    
    Suponha que temos um conjunto de dados de altura de estudantes de uma escola. 
    A densidade dos dados nos daria uma ideia de como as alturas são distribuídas entre os alunos.

    Por exemplo, se temos uma densidade de dados mais alta entre 160cm e 170cm, 
    isso significa que há uma concentração maior de estudantes com altura nessa faixa em 
    comparação com outras faixas de altura. Por outro lado, se a densidade dos dados é 
    muito baixa entre 180cm e 190cm, isso significa que há menos estudantes com altura nessa faixa.

    A densidade dos dados pode ser visualizada por meio de um gráfico de densidade, 
    que é uma representação gráfica da distribuição dos dados. Esse gráfico mostra a 
    distribuição de probabilidade dos dados ao longo de uma escala de valores. 
    As regiões com maior densidade têm uma área maior no gráfico, enquanto as regiões 
    com menor densidade têm uma área menor. A forma da curva no gráfico pode variar de 
    acordo com a distribuição dos dados e oferecer informações adicionais sobre como os dados estão distribuídos.
    '''    
    
    # Create a distribution plot using seaborn for the specified column in the 'df' dataframe, separated by stroke status
    plt.figure(figsize=(8,6))
    sns.distplot(df[df['stroke'] == 0][column_name_num], color='green', label='No Stroke')
    sns.distplot(df[df['stroke'] == 1][column_name_num], color='red', label='Stroke')
    
    # Definir propriedades de fonte
    title_font = fm.FontProperties(weight='bold', size=20)

    # Set the plot title, x-axis limit and show the plot
    plt.title(f'No Stroke vs Stroke by {column_name_num}',fontproperties=title_font)
    plt.xlabel(column_name_num)
    plt.legend()
    plt.grid(True)
    #plt.show()
    #plt.savefig(f"previsao_AVC/9-Graphics/{column_name_num}_density.png")
     
def PlotPieChart(df, column_name_cat):
    # Crie um gráfico em formato de pizza com a contagem de cada fruta
    fatias, texto, autotexto = plt.pie(df[column_name_cat].value_counts(), autopct='%.2f%%')

    # Definir propriedades de fonte
    title_font = fm.FontProperties(weight='bold', size=20)

    # Adicione rótulos nas fatias
    for fatia in fatias:
        fatia.set_label('{}'.format(fatia.get_label()))

    # Adicione legendas
    plt.legend(fatias, df[column_name_cat].unique(), loc='best')

    # Adicione título e formatação final
    plt.title(f'Distribution of {column_name_cat}',fontproperties=title_font)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"previsao_AVC/9-Graphics/{column_name_cat}_pizza.png")
    # Salvar o gráfico numericos em um arquivo
    #plt.savefig(f"previsao_AVC/9-Graphics/{column_name_cat}_pizza.png") 
    #plt.show() 
     
def main():
    # Faz a leitura do arquivo
    input_file = 'previsao_AVC/0-Datasets/AvcClear1.data' 
    names = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke'] 
    features = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']   
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names = names,      # Nome das colunas 
                     usecols = features, # Define as colunas que serão  utilizadas
                     na_values='N/A')    # Define que N/A será considerado valores ausentes

    '''
    O código print(df.describe().T)
    exibe um resumo estatístico das variáveis númericas presentes no dataframe df. 
    O método describe() é utilizado para calcular algumas estatísticas descritivas, 
    tais como contagem, média, desvio padrão, valor mínimo e máximo, 
    bem como quartis para as variáveis númericas.
    O .T no final do comando é usado para transportar a tabela, tornando mais fácil a visualização das informações.
    '''
    print(df.describe().T)
    
    '''
    O código print(df.describe(include=object).T)
    exibe um resumo estatístico das variáveis categóricas presentes no dataframe df. 
    O parâmetro include=objecté utilizado para indicar que apenas as variáveis 
    categóricas devem ser consideradas no resumo estatístico. 
    O resultado incluirá a contagem de valores únicos, 
    o valor mais frequente e a frequência do valor mais frequente.
    '''
    print(df.describe(include=object).T)
    
    '''
    O código print(f"Data has {df.shape[0]} instances and {df.shape[1] - 1} attributes.")
    exibe o número de instâncias (linhas) e atributos (colunas) presentes no dataframe df. 
    O método shapeé usado para retornar o número de linhas e colunas em um dataframe. 
    A subtração de 1 na segunda parte da string é para excluir a contagem da coluna de índice, 
    que não é considerada um atributo.
    '''
    print(f"Data has {df.shape[0]} instances and {df.shape[1] - 1} attributes.")
    
    '''
    Mostrar todos os valores e frequências de uma coluna categórica  
    Para fazer isso, você pode seguir o exemplo abaixo:
    '''
    # 1. Escolha a coluna que você deseja examinar
    for coluna in features:

        # 2. Use o método value_counts() para contar o número de ocorrências de cada valor único na coluna escolhida
        contagem = df[coluna].value_counts()

        # 3. Converta a série resultante em um novo dataframe e use o método reset_index() para redefinir o índice
        df_contagem = contagem.to_frame().reset_index()

        # 4. Renomeie as colunas para refletir o que elas representam
        df_contagem.columns = [coluna, "frequencia"]

        # 5. Use a função print() para mostrar o novo dataframe com todos os valores e frequências
        print(df_contagem)
    
    '''
    Esse código separa as colunas do dataframe df em duas listas distintas, 
    uma para colunas categóricas e outra para colunas numéricas.
    A lógica é a seguinte:
    Inicialmente, as listas cat e num são criadas vazias.
    O loop for itera sobre todas as colunas do dataframe df.
    Para cada coluna, o código verifica o tipo de dado usando o atributo dtype da coluna
    '''
    cat = []
    num = []
    for i in df.columns:
        if df[i].dtypes == object:
            cat.append(i)
        else :
            num.append(i)
    
    '''
    Este código cria um gráfico de pizza para cada coluna categórica 
    presente no dataframe df usando uma biblioteca Matplotlib.
    O loop for itera sobre todas as colunas categóricas do dataframe df, 
    armazenadas na lista cat. Para cada coluna, o código usa a função value_counts()
    dos pandas para contar o número de ocorrências de cada valor único na coluna.
    Em seguida, o código usa a função plot(kind="pie", autopct="%.2f")
    para gerar um gráfico de pizza. O parâmetro kind="pie" é usado para especificar o tipo de gráfico,
    e o parâmetro autopct="%.2f" é usado para especificar o formato das porcentagens de exibição nas fatias da pizza.
    Por fim, o código usa a função show() da biblioteca Matplotlib para exibir o gráfico na tela. 
    Em seguida, o loop itera para a próxima coluna categórica, 
    gerando um novo gráfico de pizza até que todas as colunas categóricas sejam exibidas.
    '''
    # Plotar gráficos de barras para todas as colunas categóricas
    for column_name_cat in cat:
        PlotPieChart(df, column_name_cat)  
        
     
    
    '''
    Este código cria um histograma com uma estimativa de densidade kernel 
    para cada coluna numérica presente no dataframe dfusando a 
    biblioteca Seaborn e a biblioteca Matplotlib.
    O loop for itera sobre todas as colunas numéricas do dataframe df, 
    armazenadas na lista num. Para cada coluna, o código usa a função 
    distplotda biblioteca Seaborn para traçar um histograma com uma 
    estimativa de densidade kernel.
    A função distplot é usada para visualizar a distribuição de uma única variável.
    Após a criação do histograma, o código usa a função show()
    da biblioteca Matplotlib para exibir o gráfico na tela. 
    Em seguida, o loop itera para a próxima coluna numérica, 
    gerando um novo histograma até que todas as vozes numéricas sejam exibidas.
    '''
    bins = []
    labels = []
    for column_name_num in num:   
        # Criar faixas de idade
        if column_name_num == "age":
            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
            labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90"]
        # Criar faixas de indice de massa corporal
        elif column_name_num == "bmi":
            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
        # Criar faixas de avg_glucose_level
        elif column_name_num == "avg_glucose_level":
            bins = [0, 50, 100, 150, 200, 250, 300]
            labels = ["0-50", "50-100", "100-150", "150-200", "200-250", "250-300"]

        df = create_bins(df, column_name_num, bins, labels)     
        PlotBarChart(df, column_name_num)
    
if __name__ == "__main__":
    main()
    