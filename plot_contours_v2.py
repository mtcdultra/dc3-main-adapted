import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_contours(data):

    x1_vals = np.linspace(-2, 2, 400)
    x2_vals = np.linspace(-2, 2, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    X = np.vstack([X1.ravel(), X2.ravel()]).T

    # Função objetiva
    Z = data.obj_fn(X).reshape(X1.shape)

    # Restrições
    eq_values = data.eq_resid(X,0).reshape(X1.shape)
    ineq_values = data.ineq_resid(X).reshape(X1.shape)

    # Criação do gráfico 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotando a superfície da função objetiva
    ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.7)

    # Plotando a região onde a restrição de igualdade é satisfeita (eq_resid = 0)
    # Vamos destacar onde eq_resid é aproximadamente zero, ou seja, a região de igualdade
    eq_mask = np.abs(eq_values) < 0.05  # Tolerância para mostrar a região
    ax.scatter(X1[eq_mask], X2[eq_mask], Z[eq_mask], color='red', label='Restrição de Igualdade', s=10)

    # Plotando a região onde a restrição de desigualdade é satisfeita (ineq_resid <= 0)
    # Vamos destacar onde ineq_resid é menor ou igual a zero
    ineq_mask = ineq_values <= 0
    ax.scatter(X1[ineq_mask], X2[ineq_mask], Z[ineq_mask], color='blue', label='Restrição de Desigualdade', s=10)

    # Rótulos e título
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Função Objetiva')
    ax.set_title('Função Objetiva com Restrições')

    # Legenda
    ax.legend()

    plt.show()