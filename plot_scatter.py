import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(x, y):
    # Verifica se x é uma matriz 2D
    if len(x.shape) != 2 or x.shape[1] < 1:
        raise ValueError('x deve ser uma matriz com pelo menos 1 coluna.')
    
    # Verifica se y tem o mesmo tamanho que o número de pontos em x
    if x.shape[0] != y.shape[0]:
        print(f"y ajustado para corresponder ao tamanho de x: {x.shape[0]}")
        y = y[:x.shape[0]]
    
    # Se y não for unidimensional, combinar as duas colunas de y
    if len(y.shape) > 1 and y.shape[1] > 1:
        print("y tem múltiplas colunas, combinando as duas colunas.")
        # Opção 1: Somando as colunas
        y = y[:, 0] + y[:, 1]
        # Ou Opção 2: Calculando a média das colunas
        # y = y.mean(axis=1)  # Descomente essa linha se preferir a média
    
    # Expande x para ter duas colunas (duplicando o valor de x)
    x1 = np.repeat(x[:, 0], 2)  # Replicando os valores de x para criar a coluna adicional
    x2 = np.repeat(x[:, 0], 2)  # Replicando os valores de x para criar a coluna adicional
    
    # Exibe as formas dos dados para debug
    print('x1 shape:', x1.shape)
    print('x2 shape:', x2.shape)
    print('y shape:', y.shape)
    
    # Criar gráfico de dispersão
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(x1, x2, c=y, cmap='viridis')
    
    # Adicionar os índices da matriz como rótulos nos pontos
    for i, (xi, xj) in enumerate(zip(x1, x2)):
        ax.annotate(str(i), (xi, xj), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8, color='red')
    
    plt.colorbar(scatter, label='y')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Scatter Plot with Point Labels')
    plt.show()
