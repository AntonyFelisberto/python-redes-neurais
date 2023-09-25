import numpy as np

#simulando operador or
entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([0,1,1,1])
pesos = np.array([0.0,0.0])
taxa_aprendizagem = 0.1

def step_function(soma):
    if soma >= 1:
        return 1
    return 0

def calcula_saidas(registro):
    s = registro.dot(pesos)
    return step_function(s)

def treinar():
    erros_total = 1
    while erros_total != 0:
        erros_total = 0
        for i in range(len(saidas)):
            saida_calculada = calcula_saidas(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saida_calculada)
            erros_total += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxa_aprendizagem * entradas[i][j] * erro)
                print("peso atualizado: ",str(pesos[j]))
        print("total erros: ",str(erros_total))

treinar()
print("rede treinada")
print(calcula_saidas(entradas[0]))
print(calcula_saidas(entradas[1]))
print(calcula_saidas(entradas[2]))
print(calcula_saidas(entradas[3]))
