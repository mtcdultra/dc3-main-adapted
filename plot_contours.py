import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Função para plotar os contornos
def plot_contours(data):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Linhas para criar o grid de contornos
    x1 = np.linspace(-2, 4, 400)
    x2 = np.linspace(-2, 4, 400)
    
    x1_ = data.trainX[:, 0]
    x2_ = data.trainX[:, 1]
    
    # Criando uma grade de valores de x1 e x2
    x1_vec, x2_vec = np.meshgrid(x1, x2)

    # Calculando a função objetivo (passando x1_vec e x2_vec empacotados em X)
    X_vec = np.column_stack([x1_vec.ravel(), x2_vec.ravel()])
    z = data.obj_fn(X_vec)
    z = z.reshape(x1_vec.shape)  # Reshape para manter as dimensões 2D

    # Calculando os resíduos das equações (passando X_vec)
    eq_resid_vals = data.eq_resid(X_vec, 0)
    eq_resid_vals = eq_resid_vals.reshape(x1_vec.shape)  # Reshape para 2D

    # Calculando os resíduos de desigualdade (passando X_vec)
    ineq_resid_vals = data.ineq_resid(X_vec)    
    ineq_resid_vals = ineq_resid_vals.reshape(x1_vec.shape)  # Reshape para 2D

    # Função objetivo para o contorno (passando X_vec)
    obj_fn_plot_vals = data.obj_fn_plot(X_vec, 1, 1)  # rh=1, rg=1
    obj_fn_plot_vals = obj_fn_plot_vals.reshape(x1_vec.shape)  # Reshape para 2D

    print(f"Shape of obj_fn_plot_vals: {obj_fn_plot_vals.shape}")

    # Plotando o contorno da função objetivo
    cp = ax.contour(
        x1_vec, x2_vec, z, levels=np.linspace(-5, 200, 300), cmap="viridis"
    )
    ax.clabel(cp, fmt="%2.2f", inline=True)

    # Contornos para os resíduos de equações (vermelho) e desigualdades (azul)
    cg1 = ax.contour(x1_vec, x2_vec, eq_resid_vals, levels=[0], colors="red", linewidths=2)
    cg2 = ax.contour(x1_vec, x2_vec, ineq_resid_vals, levels=[0], colors="blue", linewidths=2)

    # Contorno da função objetivo (amarelo)
    cg3 = ax.contour(
        x1_vec, x2_vec, obj_fn_plot_vals, levels=[0], colors="yellow", linewidths=2
    )

    # Plotando a trajetória (se fornecida)
    #print(trajectory)
    
    
        # Adicionando título e labels
    plt.title("Contours of the objective function")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)

    # Adicionando legenda
    plt.legend()

    plt.show()

