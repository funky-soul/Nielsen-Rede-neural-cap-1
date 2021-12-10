#!/usr/bin/python3
"""Cabeçalho
  +-----------------------------------------------------------------------------------+
  | Autor: [tradução do código original de Michael Nielsen]                           |
  | Data: 09/12/2021                                                                  |
  | Universidade Federal de Viçosa (UFV)                                              |
  | Departamento de física (DPF)                                                      |
  | Matéria FIS 493 - Tópicos especiais (redes neurais)                               |
  +-----------------------------------------------------------------------------------+
  """

""" O que este código faz:
  O "carregador_mnist" é uma biblioteca que carrega os dados de imagem
  do MNIST. Para detalhes das estruturas de dados que são retornados,
  veja a os comentários para "carregador_de_dados" e
  "empacotador_de_dados_carregados". Na prática, "empacotador_de_dados_carregados"
  é a função usualmente chamada pelo nosso código de rede neural.
  """

# Bibliotecas
  # Bibliotecas padrões
import pickle                     # Cria representações de obj python serializadas.
import gzip                       # Torna possível a leitura do arquivo zippado.

  #Bibliotecas de terceiros:
import numpy as np                # Facilita a criação de matrizes e tem maior velocidade.

""" Comentário (carregador_de_dados):
    Retorna os dados do MNIST como uma tuple contendo os dados de treino,
  os dados de validação, e o resto dos dados.
    O "dados_de_treino" é retornado como uma tuple com duas entradas.
  A primeira entrada contém as imagens de treino. Mais especificamente,
  esta entrada é uma matriz do numpy (ndarray) com 50.000 (cinquenta mil) entradas.
  Cada entrada é, por sua vez, uma matriz numpy (ndarray) com 784 valores, representando
  os 28*28 = 784 pixels em uma única imagem MNIST.
    A segunda entrada no tuple "dados_de_treino" é uma matriz numpy (ndarray) contendo
  50.000 (cinquenta mil) entradas. Essas entradas são apenas os valores de digitos 
  (0, ..., 9) para as imagens correspondentes contidas no tuple da primeira entrada.
    Os "dados_de_validacao" e "dados_de_teste" são similares, com execeção que cada
  um contém apenas 10.000 (dez mil) imagens.
    Este é um bom formato de dados, mas para o uso em redes neurais é útil se
  modificarmos um pouco o formato dos "dados_de_treino". Isso é feito através da função
  empacotadora (wrapper) "empacotador_de_dados_carregados", veja isso abaixo.
  """
def carregador_de_dados():
  f = gzip.open('mnist.pkl.gz','rb') #Lê arquivo tipo binário e zippado.

  # A linha abaixo gera uma estruturação dos dados do arq nas variáveis.
  dados_de_treino, dados_de_validacao, dados_de_teste = pickle.load(f, encoding = 'latin1')
  
  f.close() # Evita que o arquivo seja corrompido.

  return (dados_de_treino, dados_de_validacao, dados_de_teste)

""" Comentário (empacotador_de_dados_carregados):
    Retorna a tuple contendo "(dados_de_treino, dados_de_validacao, dados_de_teste)".
  Baseado no "dados_carregados", mas o formato é mais conveniente para o uso em nossa
  implementação de redes neurais. Em particular, "dados_de_treino" é uma lista contendo
  50.000 (cinquenta mil) tuplas 2-D "(x,y)". "x" é uma matriz numpy (ndarray) de 784
  dimensões, contendo a entrada de imagem, e "y" é a classificação correspondente, i.e.,
  os valores dos dígitos (inteiros) correspondente ao "x".
    "dados_de_validacao" e "dados_de_teste" são listas contendo 10.000 (dez mil) tuplas
  de 2 dimensões "(x, y)". Em cada caso, "x" é uma matriz numpy (ndarray) contendo a 
  entrada de dados das imagens, e "y" é a classificação correspondente, i.e., os valores
  dos digitos (inteiros) correspondentes ao "x".
    Obiviamente, isso significa que nós estamos usando formatos ligeramente diferentes
  para os dados de treino e para os dados de validação/teste. Esses formatos mostraram-se
  ser os mais convenientes para o uso em nosso código de rede neural.
  """
def empacotador_de_dados_carregados():

  # As estruturas geradas na função "carregador_de_dados" são jogadas nas variáveis abaixo:
  d_treino, d_validacao, d_teste = carregador_de_dados() # Cada variável é 2-D

  entradas_de_treino             = [np.reshape(x, (784, 1)) for x in d_treino[0]]
  resultados_de_treino           = [resultado_vetorizado(y) for y in d_treino[1]]
  dados_de_treino                = zip(entradas_de_treino, resultados_de_treino)

  entradas_de_validacao          = [np.reshape(x, (784, 1)) for x in d_validacao[0]]
  dados_de_validacao             = zip(entradas_de_validacao, d_validacao[1])

  entradas_de_teste              = [np.reshape(x, (784, 1)) for x in d_teste[0]]
  dados_de_teste                 = zip(entradas_de_teste, d_teste[1])

  return (dados_de_treino, dados_de_validacao, dados_de_teste)

""" Comentário (resultado_vetorizado):
  Retorna um vetor unitário de 10 dimensões, com 1. nas
  j-posições e 0. nas outras posições. Isso é usado para 
  converter um dígito (0, ..., 9) em uma saída desejável 
  correspondente para a rede neural.
  """ 
def resultado_vetorizado(j):
  vetor = np.zeros((10, 1))
  vetor[j] = 1.

  return vetor