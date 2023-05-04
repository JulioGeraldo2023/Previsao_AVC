from math import sqrt

# funçao de calculo de distancia
def distancia_euclideana(vet1, vet2):
    distancia = 0
    for i in range(len(vet1)-1):
        distancia += (vet1[i] - vet2[i])**2
    distancia = sqrt(distancia)
    return distancia

#funçao retorna k vizinhos mais proximos
def retorna_vizinhos(base_treinamento, amostra_test, numero_vizinhos):
    distancias = list()
    for linha_tre in base_treinamento:
        dist = distancia_euclideana(amostra_test, linha_tre)
        distancias.append((linha_tre, dist))
    #ordenaçao das distancias de forma crescente
    distancias.sort(key=lambda tup: tup[1])
    #distancias = sorted(distancias, key=lambda x:float(x))
    #retorna os vizinhos mais proximos
    vizinhos = list()
    for i in range(numero_vizinhos):
        vizinhos.append(distancias[i][0])
    return vizinhos
    
#funçao de prediçao
def classificaçao(base_treinamento, amostra_test, numero_vizinhos):
    vizinhos = retorna_vizinhos(base_treinamento, amostra_test, numero_vizinhos)
    print(vizinhos)
    rotulos = [v[-1] for v in vizinhos]
    predicao = max(set(rotulos),key=rotulos.count)
    return predicao

dataset =   [[2.1, 2.3, 0],
            [2.2, 2.5, 0],
            [2.3, 2.4, 0],
            [4.4, 2.2, 0],
            [4.5, 2.3, 1],
            [5.6, 2.3, 1],
            [6.7, -2.3, 1],
            [7.4, 2.3, 1]]

#amostra = [2.1, 2.3, 0]
amostra = [4.4, 2.3, 0]
#vizinhos = retorna_vizinhos(dataset, amostra, 3)
predicao = classificaçao(dataset, amostra, 3)
print('Rotulos %d\nPredição %d ' % (amostra[-1], predicao))




'''
amostra = dataset[0] # [7.5, 2.3, 0]
for linha in dataset:
    distancia = distancia_euclideana(amostra, linha)
    print(distancia)


for v in vizinhos:
    print(v)
'''
