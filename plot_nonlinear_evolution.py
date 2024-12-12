import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_nonlinear_evolution(data, y_new_history, point_index=0):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Linhas para criar o grid de contornos
    x1 = np.linspace(-2, 4, 400)
    x2 = np.linspace(-2, 4, 400)

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

    # Plotando o contorno da função objetivo
    cp = ax.contour(
        x1_vec, x2_vec, z, levels=np.linspace(-5, 200, 300), cmap="viridis"
    )
    ax.clabel(cp, fmt="%2.2f", inline=True)

    # Contornos para os resíduos de equações (vermelho) e desigualdades (azul)
    cg1 = ax.contour(x1_vec, x2_vec, eq_resid_vals, levels=[0], colors="red", linewidths=2)
    cg2 = ax.contour(x1_vec, x2_vec, ineq_resid_vals, levels=[0], colors="blue", linewidths=2)

    # Ponto inicial (exemplo: primeiro ponto de treinamento)
    x_init = data.trainX[0, 0]  # Pega o valor de x1 no ponto inicial
    x_init_2 = data.trainX[0, 1]  # Pega o valor de x2 no ponto inicial

    # Plotando o ponto inicial em verde
    ax.scatter(x_init, x_init_2, color='green', label='Ponto Inicial')

    # Inicializando 'scatter' para garantir que sempre tenha um valor antes de usar
    scatter = None

    # Verifica se o ponto index está dentro do intervalo de y_new_history
    
    if point_index < len(y_new_history):
        y_new = y_new_history[point_index]
        
        # Inicializa uma lista para armazenar a trajetória de pontos
        trajectory_x1 = [x_init]  # Começa com o ponto inicial
        trajectory_x2 = [x_init_2]

        # A seguir, iteramos apenas uma vez por ponto de evolução
        print('$$$$$ ', len(y_new))
        
        for i in range(len(y_new)):
            # Garantir que estamos pegando apenas 1 ponto por época
            trajectory_x1.append(y_new[i][0])  # Adiciona x1
            trajectory_x2.append(y_new[i][1])  # Adiciona x2

        # Plotando a linha que conecta todos os pontos evoluídos
        ax.plot(trajectory_x1, trajectory_x2, label=f'Evolução no Ponto {point_index+1}', linestyle='-', linewidth=2)

        # Plotando os pontos de y_new sobre os contornos, incluindo intermediários
        scatter = ax.scatter(trajectory_x1[1:], trajectory_x2[1:], c=data.obj_fn(np.array([trajectory_x1[1:], trajectory_x2[1:]]).T), cmap='viridis', edgecolors='black', label=f'Pontos {point_index+1}')
        
        # Destacando o ponto final em vermelho
        ax.scatter(trajectory_x1[-1], trajectory_x2[-1], color='red', label='Ponto Final')
        
        # Calculando e mostrando as distâncias percorridas entre os pontos
#        for i in range(1, len(trajectory_x1)):
#            distance = np.sqrt((trajectory_x1[i] - trajectory_x1[i-1])**2 + (trajectory_x2#[i] - trajectory_x2[i-1])**2)
#            print(f"Distância percorrida entre o ponto {i} e o ponto {i+1}: {distance:.2f}")
        
    else:
        print("Índice de ponto inválido!")

    # Adicionando barra de cores apenas se 'scatter' foi inicializado
    if scatter is not None:
        plt.colorbar(scatter, label="Objetivo")

    # Adicionando título e labels
    plt.title("Contours of the objective function with model output points")    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)

    # Adicionando legenda
    plt.legend()

    # Exibindo o gráfico
    plt.show()