#pip install http://github.com/pybrain/pybrain/archive/0.3.3.zip
from pybrain.structure import FeedForwardNetwork, LinearLayer,SigmoidLayer,BiasUnit,FullConnection


rede = FeedForwardNetwork()

camada_entrada = LinearLayer(2)
camada_oculta = SigmoidLayer(3)
camada_saida = SigmoidLayer(1)

bias_um = BiasUnit()
bias_dois = BiasUnit()

rede.addModule(camada_entrada)
rede.addModule(camada_oculta)
rede.addModule(camada_saida)
rede.addModule(bias_um)
rede.addModule(bias_dois)

entrada_oculta = FullConnection(camada_entrada,camada_oculta)
oculta_saida = FullConnection(camada_oculta,camada_saida)
bias_oculta = FullConnection(bias_um,camada_oculta)
bias_saida = FullConnection(bias_dois,camada_saida)



rede.sortModules()

print(rede)
print(entrada_oculta.params)
print(oculta_saida.params)
print(bias_oculta.params)
print(bias_saida.params)
