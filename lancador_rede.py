#!/usr/bin/python3

#Primeiro chamamos a bbt responsável por ler o arquivo MNIST
import carregador_mnist
#Informamos as variáveis que queremos chamar da função (des)empacotador de dados
dados_de_treino, dados_de_validacao, dados_de_teste = (
                                        carregador_mnist.empacotador_de_dados_carregados())
dados_de_treino = list(dados_de_treino)

#Importamos a bbt responsável pela rede neural
import rede
# Tamanho e quantidade de camadas da rede. Primeiro número são os inputs, a última é o 
# output. O(s) do meio - quantos quiser -, são as camadas ocultas.
net = rede.Rede((784, 15, 10))

# Dados de treino(tuple); Épocas(int); Tamanho do mini-lote(int); Taxa de aprendizado(float); 
# dados de teste (tuple - opcional)
net.GED(dados_de_treino, 5, 25, 32., dados_de_teste=dados_de_teste)
