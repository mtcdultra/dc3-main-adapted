import matplotlib.pyplot as plt
import numpy as np

def plot_nonconvex(x, y):
    # Verifica se x é uma matriz 2D e possui pelo menos 2 colunas
    if len(x.shape) != 2 or x.shape[1] < 2:
        # Expande x para ter duas colunas, caso necessário (duplicando a primeira coluna)
        print("x tem apenas uma coluna, expandindo para duas.")
        x = np.repeat(x, 2, axis=1)
    
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
    
    # Atribui x1 e x2 a partir de x
    x1 = x[:, 0]
    x2 = x[:, 1]
    
    print('x1 shape:', x1.shape)
    print('x2 shape:', x2.shape)
    print('y shape:', y.shape)
    
    # Criar o gráfico de dispersão
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(x1, x2, c=y, cmap='viridis')

    # Adicionar os índices da matriz como rótulos nos pontos
    for i, (xi, xj) in enumerate(zip(x1, x2)):
        ax.annotate(str(i), (xi, xj), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8, color='red')

    # Adicionar barra de cores
    plt.colorbar(scatter, label='y')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Scatter Plot with Contours and Point Labels - Quadratic problem')

    # Criando o grid de contornos
    x1_grid = np.linspace(min(x1)-1, max(x1)+1, 100)
    x2_grid = np.linspace(min(x2)-1, max(x2)+1, 100)
    x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)

    # Calculando a função objetivo ou qualquer outra função para o contorno
    Z = np.sin(x1_grid) + np.cos(x2_grid)  # Exemplo de função para o contorno

    # Plotando os contornos
    contour = ax.contour(x1_grid, x2_grid, Z, levels=10, cmap='coolwarm')
    
    # Exibindo o gráfico
    plt.show()
