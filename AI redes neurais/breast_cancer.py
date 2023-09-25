import numpy as np
from sklearn import datasets

base = datasets.load_breast_cancer()
entradas = base.data
valores_saida = base.target
saidas = np.empty([569,1],dtype=int)

for i in range(569):
    saidas[i] = valores_saida[i]

pesos_zero = 2*np.random.random((30,3)) - 1
pesos_um = 2*np.random.random((3,1)) - 1

epocas = 10000
taxa_aprendizagem = 0.6
momento = 1

def sigmoid_function(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoid_derivada(sig):
    return sig * (1 - sig)

for j in range(epocas):
    camada_entradas = entradas

    soma_sinapse = np.dot(camada_entradas,pesos_zero)
    camada_oculta = sigmoid_function(soma_sinapse)

    soma_sinapse_um = np.dot(camada_oculta,pesos_um)
    camada_saida = sigmoid_function(soma_sinapse_um)

    erro_camada_saida = saidas - camada_saida
    media_absoluta = np.mean(np.abs(erro_camada_saida))
    print("erro: ",str(media_absoluta))

    derivada_saida = sigmoid_derivada(camada_saida)
    delta_saida = erro_camada_saida * derivada_saida

    pesos_um_transposta = pesos_um.T
    delta_saida_x_peso = delta_saida.dot(pesos_um_transposta)
    delta_camada_oculta = delta_saida_x_peso * sigmoid_derivada(camada_oculta)

    camada_oculta_transposta = camada_oculta.T
    pesos_novo_um = camada_oculta_transposta.dot(delta_saida)
    pesos_um = (pesos_um * momento) + (pesos_novo_um * taxa_aprendizagem)

    camada_entradas_transposta = camada_entradas.T
    pesos_novo_zero = camada_entradas_transposta.dot(delta_camada_oculta)
    pesos_zero = (pesos_zero * momento) + (pesos_novo_zero * taxa_aprendizagem)