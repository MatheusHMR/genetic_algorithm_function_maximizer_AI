import numpy as np
import math
import random

def get_n_bits(bounds, precision):
    """Calcula quantos bits são necessários para cada variável."""
    bits = []
    for low, high in bounds:
        span = int((high - low) * (10**precision))
        bits.append(math.ceil(math.log2(span + 1)))
    return bits

def real_to_bin(x, low, precision, n_bits):
    """Converte valor real x para inteiro com precisão e depois para string binária."""
    factor = 10**precision
    x_int = int(round((x - low) * factor))
    return format(x_int, f'0{n_bits}b')

def bin_to_real(bstr, low, precision):
    """Converte string binária de volta para real, aplicando offset e precisão."""
    factor = 10**precision
    return int(bstr, 2) / factor + low

def single_point_binary_crossover(p1, p2, bounds, precision):
    """
    Realiza o cruzamento de ponto único entre dois indivíduos na representação binária.
    
    :param p1: Primeiro pai (array de valores reais)
    :param p2: Segundo pai (array de valores reais)
    :param bounds: Lista de tuplas com os limites (min, max) de cada variável
    :param precision: Precisão decimal para a codificação binária
    :return: Filho resultante do cruzamento (array de valores reais)
    """
    # Determina o número de bits necessários para cada variável
    bits_list = get_n_bits(bounds, precision)

    print(f"x1 com bits: {bits_list[0]}")
    print(f"x2 com bits: {bits_list[1]}")
    
    # 1) Codifica cada indivíduo para representação binária
    b1 = ''
    b2 = ''
    for i in range(len(p1)):
        xp1, xp2 = p1[i], p2[i]
        low, _ = bounds[i]
        n_bits = bits_list[i]
        b1 += real_to_bin(xp1, low, precision, n_bits)
        b2 += real_to_bin(xp2, low, precision, n_bits)

    print("b1: ", b1)
    print("b2: ", b2)
    # 2) Escolhe um ponto de corte aleatório
    point = random.randint(1, len(b1) - 1)
    
    # 3) Realiza o cruzamento
    child_bin = b1[:point] + b2[point:]
    
    # 4) Decodifica o filho de volta para valores reais
    child = []
    idx = 0 # Começa na posição 0
    for i, (low, _) in enumerate(bounds):
        n_bits = bits_list[i]
        snippet = child_bin[idx:idx+n_bits]
        child.append(bin_to_real(snippet, low, precision))
        idx += n_bits

    # Fazer dessa forma habilita que o crossover ajuste o domínio para o problema independentemente da quantidade de variáveis
    lower_bounds = np.array([low for low, _ in bounds]) 
    upper_bounds = np.array([high for _, high in bounds])

    return np.clip(np.array(child), lower_bounds, upper_bounds)

def double_point_binary_crossover(p1, p2, bounds, precision):
    """
    Realiza o cruzamento de dois pontos entre dois indivíduos na representação binária.
    
    :param p1: Primeiro pai (array de valores reais)
    :param p2: Segundo pai (array de valores reais)
    :param bounds: Lista de tuplas com os limites (min, max) de cada variável
    :param precision: Precisão decimal para a codificação binária
    :return: Filho resultante do cruzamento (array de valores reais)
    """
    # Determina o número de bits necessários para cada variável
    bits_list = get_n_bits(bounds, precision)
    
    # 1) Codifica cada indivíduo para representação binária
    b1 = ''
    b2 = ''
    for i in range(len(p1)):
        xp1, xp2 = p1[i], p2[i]
        low, _ = bounds[i]
        n_bits = bits_list[i]
        b1 += real_to_bin(xp1, low, precision, n_bits)
        b2 += real_to_bin(xp2, low, precision, n_bits)
    
    # 2) Escolhe dois pontos de corte aleatórios
    a, b = sorted(random.sample(range(1, len(b1)), 2)) # Ordena os pontos para que a seja o menor
    
    # 3) Realiza o cruzamento
    child_bin = b1[:a] + b2[a:b] + b1[b:]

    # Exemplo caso a = 2 e b = 5 e len(b1) = 8
    # b1 = 11100011
    # b2 = 00111100
    # child_bin = 11|111|011
    #             ^^|^^^|^^^
    #             b1|b2 |b1

    # 4) Decodifica o filho de volta para valores reais
    child = []
    idx = 0
    for i, (low, _) in enumerate(bounds):
        n_bits = bits_list[i]
        snippet = child_bin[idx:idx+n_bits]
        child.append(bin_to_real(snippet, low, precision))
        idx += n_bits

    lower_bounds = np.array([low for low, _ in bounds])
    upper_bounds = np.array([high for _, high in bounds])

    return np.clip(np.array(child), lower_bounds, upper_bounds)
        
# Exemplo de uso:
bounds = [(-3.1, 12.1), (4.1, 5.8)]   # domínio de x1 e x2
precision = 1                # 1 casa decimal
p1 = np.array([5.1, 4.7])
p2 = np.array([7.4, 5.7])
filho_sp = single_point_binary_crossover(p1, p2, bounds, precision)
filho_dp = double_point_binary_crossover(p1, p2, bounds, precision)
print("Single-point child:", filho_sp)
print("Double-point child:", filho_dp)
