import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

def plot_nonlinear_evolution_3d(data, y_new_history, point_index=0):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Linhas para criar o grid de contornos
    x1 = np.linspace(-2, 4, 400)
    x2 = np.linspace(-2, 4, 400)

    # Criando uma grade de valores de x1 e x2
    x1_vec, x2_vec = np.meshgrid(x1, x2)

    # Calculando a função objetivo (passando x1_vec e x2_vec empacotados em X)
    X_vec = np.column_stack([x1_vec.ravel(), x2_vec.ravel()])
    z = data.obj_fn(X_vec)
    z = z.reshape(x1_vec.shape)  # Reshape para manter as dimensões 2D

    # Plotando a superfície da função objetivo
    surf = ax.plot_surface(x1_vec, x2_vec, z, cmap="viridis", alpha=0.8, edgecolor='none')

    # Adicionando barra de cores para a superfície
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Objetivo")

    # Ponto inicial
    x_init = data.trainX[0, 0]
    x_init_2 = data.trainX[0, 1]

    # Verifica se o ponto index está dentro do intervalo de y_new_history
    if point_index < len(y_new_history):
        y_new = y_new_history[point_index]
        
        # Inicializa listas para a trajetória
        trajectory_x1 = [x_init]  # Começa com o ponto inicial
        trajectory_x2 = [x_init_2]
        trajectory_z = [data.obj_fn(np.array([[x_init, x_init_2]]))[0]]  # Valor inicial do objetivo

        for i in range(len(y_new)):
            trajectory_x1.append(y_new[i][0])
            trajectory_x2.append(y_new[i][1])
            trajectory_z.append(data.obj_fn(np.array([[y_new[i][0], y_new[i][1]]]))[0])

        # Plotando a trajetória em 3D
        ax.plot(
            trajectory_x1, trajectory_x2, trajectory_z,
            label=f"Evolução no Ponto {point_index + 1}",
            linestyle="-", linewidth=2, color="red"
        )

        # Marcando os pontos intermediários
        ax.scatter(
            trajectory_x1[1:], trajectory_x2[1:], trajectory_z[1:],
            c=trajectory_z[1:], cmap="viridis", edgecolor="black"
        )

        # Destacando o ponto final e colocando legenda apenas para o ponto final
        ax.scatter(
            trajectory_x1[-1], trajectory_x2[-1], trajectory_z[-1],
            color="red", s=100, label="Ponto Final"
        )
    else:
        print("Índice de ponto inválido!")

    # Adicionando título e labels
    ax.set_title("Evolução da Função Objetivo em 3D", fontsize=14)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Objetivo")

    # Adicionando legenda (somente para o ponto final)
    ax.legend()

    # Exibindo o gráfico
    plt.show()