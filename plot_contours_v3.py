import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_contours(data, points=None):
        
    x1_vals = np.linspace(-1, 4, 400)
    x2_vals = np.linspace(-1, 4, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    X = np.vstack([X1.ravel(), X2.ravel()]).T

    # Cálculo da função objetiva nos pontos
    Z = data.obj_fn(X).reshape(X1.shape)

    # Cálculo das restrições
    eq_values = data.eq_resid(X, 0).reshape(X1.shape)
    ineq_values = data.ineq_resid(X).reshape(X1.shape)

    # Criando o gráfico
    plt.figure(figsize=(8, 6))

    # Plotando as curvas de contorno da função objetiva
    cp = plt.contour(X1, X2, Z, levels=20, cmap='twilight', alpha=0.7)

    # Adicionando as curvas de contorno para a restrição de igualdade (eq_resid = 0)
    eq_mask = np.abs(eq_values) < 0.05  # Tolerância para mostrar a região
    plt.contour(X1, X2, eq_values, levels=[0], colors='red', linewidths=2, label="Restrição de Igualdade")

    # Plotando as curvas de contorno para a restrição de desigualdade (ineq_resid <= 0)
    ineq_mask = ineq_values <= 0
    plt.contour(X1, X2, ineq_values, levels=[0], colors='blue', linewidths=2, linestyles='dashed', label="Restrição de Desigualdade")

    # Se pontos forem fornecidos, plotá-los no gráfico
    if points is not None:
        points = np.array(points)
        plt.plot(points[:, 0], points[:, 1], 'k-', label='Trajetória do Treinamento', alpha=0.6)
        plt.scatter(points[0, 0], points[0, 1], color='green', label='Ponto Inicial', zorder=5)
        plt.scatter(points[-1, 0], points[-1, 1], color='orange', label='Ponto Final', zorder=5)

    # Adicionando título e rótulos
    plt.title('Função Objetiva com Restrições')
    plt.xlabel('x1')
    plt.ylabel('x2')

    # Adicionando a barra de cores para a função objetiva
    plt.colorbar(cp)

    # Adicionando legenda
    plt.legend()

    plt.show()