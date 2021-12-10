"""Cabeçalho
  +-----------------------------------------------------------------------------------+
  | Autor: [tradução do código original de Michael Nielsen]                           |
  | Data: 08/12/2021                                                                  |
  | Universidade Federal de Viçosa (UFV)                                              |
  | Departamento de física (DPF)                                                      |
  | Matéria FIS 493 - Tópicos especiais (redes neurais)                               |
  +-----------------------------------------------------------------------------------+
  """

"""O que este código é:
  Este código é um módulo que implementa o algorismo de aprendizado gradiente estocastico
  decrescente para uma rede neural de alimentação direta. Os gradientes são calculados 
  usando retropropagação. Note que o autor original (Michael Nielsen) focou em tornar o 
  código simples, facilmente legivel e facilmente modificável. Não é otimizado e omite 
  muitos recursos desejáveis.
  """

#Bibliotecas:
  #Bibliotecas padrões:
import random               # Gerador de variáveis aletórias (bytes, int, ou sequencias)

  #Bibliotecas de terceiros:
import numpy as np          # Facilita a criação de matrizes e tem maior velocidade.

class Rede(object):
  
  """Comentário (__init__):
    A lista "tamanhos" contém o número de neurônios nas respectivas camadas da rede. Por 
    exemplo, se a lista for [2, 3, 1] então isso seria uma rede de 3 camadas, com a primeira
    camada possuindo 2 neurônios, a segunda camada 3 neurônios e a terceira camada possuindo
    um neurônio. As propensões e pesos são inicializados aleatoriamente, usando distribuição
    gaussiana com média 0 e variância 1. Note que assumi-se que a primeira camada é a camada
    de entrada e, por convenção, nos não colocamos nenhuma propensão para esses neurônios, 
    desde que as propensões são apenas usadas para computar as saidas das camadas seguintes.
    """
  def __init__(auto, tamanhos):
    
    auto.num_camadas = len(tamanhos)
    auto.tamanhos    = tamanhos
    auto.propensoes  = [np.random.randn(y, 1) for y in tamanhos[1:]]
    auto.pesos       = [np.random.randn(y, x) for x, y in zip (tamanhos[:-1],tamanhos[1:])]


  # Retorna a saida da rede se "a" for uma entrada.
  def alimentacao_direta(auto, entrada):   #feedforward
    
    for propensao, peso in zip(auto.propensoes, auto.pesos):
      entrada = sigmoide(np.dot(peso, entrada) + propensao)
    
    return entrada



  """Comentário (Gradiente Estocastico Decrescente - GED):
    Treinaremos a rede neural usando mini-lotes de gradientes decrescentes estocástico. O 
    "dados_de_treino" é uma lista de tuples "(x,y)" representando o treino das entradas e 
    saídas desejadas. Caso "dados_de_teste" seja provido, então a rede irá avaliar com os 
    dados de teste depois de cada época, e o progresso parcial será printado. Isso é útil 
    para acompanhar o progresso, mas torna mais lento o processo.
  
    OBS: testar sem o print depois
    OBS 2: "gradiente estocastico decrescente" será abreviado como GED
    """
  def GED(auto, dados_de_treino, epocas, mini_lote_tamanho, taxa_de_aprendizado, 
                                                                      dados_de_teste=None):
  
    dados_de_treino = list(dados_de_treino)
    tam             = len(dados_de_treino)
  
    if dados_de_teste:
      dados_de_teste = list(dados_de_teste)
      tam_teste      = len(dados_de_teste)
    
    for j in range(epocas):
      random.shuffle(dados_de_treino)
      mini_lotes = [dados_de_treino[k:k+mini_lote_tamanho] for k in range(0, tam, 
                                                                        mini_lote_tamanho)]

      for mini_lote in mini_lotes:
        auto.atualiza_mini_lote(mini_lote, taxa_de_aprendizado)

      if dados_de_teste:
        print("Época {}: {} / {}".format(j, auto.avaliar(dados_de_teste), tam_teste))
      else:
        print("Época {} completa".format(j))


  """Comentário (atualiza_mini_lote):
    Atualiza os pesos e propensões da rede ao aplicar o gradiente decrescente estocástico 
    usando retropropagação para um único mini lote. O "mini_lote" é uma lista de tuples 
    "(x,y)".
    """
  def atualiza_mini_lote(auto, mini_lote, taxa_de_aprendizado):

    nabla_propensao = [np.zeros(propensao.shape) for propensao in auto.propensoes]
    nabla_pesos     = [np.zeros(peso.shape) for peso in auto.pesos]
    
    for x, y in mini_lote:
      delta_nabla_propensao, delta_nabla_pesos = auto.retropropagacao(x, y)

      nabla_propensao = [nabla_prop + delta_nabla_prop for nabla_prop,
                          delta_nabla_prop in zip (nabla_propensao, delta_nabla_propensao)]

      nabla_pesos = [nabla_pes + delta_nabla_pes for nabla_pes, delta_nabla_pes
                                                  in zip (nabla_pesos, delta_nabla_pesos)]

    auto.propensoes = [propensao-(taxa_de_aprendizado/len(mini_lote))*nova_propensao
                    for propensao, nova_propensao in zip(auto.propensoes, nabla_propensao)]

    auto.pesos  = [peso-(taxa_de_aprendizado/len(mini_lote))*novo_peso for peso,
                                                novo_peso in zip(auto.pesos, nabla_pesos)]
  


  """Comentário (retropropagacao):
    Retorna uma tuple "(nabla_propensao, nabla_peso)" representando o
    gradiente para a função de custo C_x. "nabla_propensao" e "nabla_peso"
    são listas de camada-por-camada de matrizes numpy, similar com o
    "auto.propensoes" e "auto.pesos".
    """
  def retropropagacao(auto, x, y):
    
    nabla_propensao = [np.zeros(propensao.shape) for propensao in auto.propensoes]
    nabla_pesos     = [np.zeros(peso.shape) for peso in auto.pesos]

    #Alimentação direta (feedfoward)
    ativacao  = x
    ativacoes = [x]     # É uma lista para armazenar todas as ativações, camada por camada.
    zs        = []      # Lista para guardar todos os z's para a função sigmóide
    
    for propensao, peso in zip (auto.propensoes, auto.pesos):
      z        = np.dot(peso, ativacao)+propensao
      zs.append(z)
      ativacao = sigmoide(z)
      ativacoes.append(ativacao)

    # Passe retrógrafo (backward pass)
    delta               = auto.custo_derivado(ativacoes[-1], y) * sigmoide_primo(zs[-1])
    nabla_propensao[-1] = delta
    nabla_pesos[-1]     = np.dot(delta,ativacoes[-2].transpose())
    
    """Comentário (passe retrógrafo):
      Note que a variável l no loop abaixo é usado de forma
      diferente a notação do Capítulo 2 do livro. Aqui,
      l = 1 significa a última camada de neurônios, l = 2
      representa a penúltima camada, e por aí vai. É uma renumeração
      do esquema do livro, usado aqui para tomar vantagem do fato
      que o python permmite usar índices negativos em listas.
      """
    for l in range(2, auto.num_camadas):
      z                   = zs[-l]
      sp                  = sigmoide_primo(z)
      delta               = np.dot(auto.pesos[-l+1].transpose(), delta) * sp 
      nabla_propensao[-l] = delta
      nabla_pesos[-l]     = np.dot(delta, ativacoes[-l-1].transpose())
    
    return (nabla_propensao, nabla_pesos)


  """Comentário (avaliar):
    Retorna o número de cada entrada de teste que tenha obtido 
    o resultado correto de saída da rede neural. Note que 
    as saídas da rede neural são consideradasa como sendo
    o índice de qualquer neurônio na camada final que tenha
    a maior ativação. 
    """
  def avaliar(auto, dados_de_teste):
    resultado_teste = [(np.argmax(auto.alimentacao_direta(x)), y) for (x,y) in 
                                                                            dados_de_teste]
    
    return sum(int( x == y ) for (x, y) in resultado_teste)


  """Comentário (custo_derivado):
    Retorna o vetor das derivadas parciais {\partial (C_x) /
    \partial (a)} para as saídas de ativação.
    """
  def custo_derivado(auto, saidas_ativadas, y):
    return (saidas_ativadas-y)


#Funções diversas:
def sigmoide(z):
  return 1./(1.+np.exp(-z))

def sigmoide_primo(z):
  return sigmoide(z)*(1-sigmoide(z))