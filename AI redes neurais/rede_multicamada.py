import numpy as np

entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([[0],[1],[1],[0]])

#DADOS PARA TESTES
pesos_zero = np.array([[-0.424,-0.740,-0.961],
                  [0.358,-0.577,-0.469]])

pesos_um = np.array([[-0.017],[-0.893],[0.148]])

#DADOS RANDOMICOS PARA VALIDAÇÃO
pesos_zero = 2*np.random.random((2,3)) - 1
pesos_um = 2*np.random.random((3,1)) - 1

epocas = 10000
taxa_aprendizagem = 0.6
momento = 1

def sigmoid_function(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoid_derivada(sig):
    return sig * (1 - sig)
    
def testes():
    a = sigmoid_function(0.5)
    b = sigmoid_derivada(a)
    print(b)
    print(np.exp(1))
    print(sigmoid_function(50))

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