"""
Code for assignment1, exercise 2 of Statistical Machine Learning

Author: Joris van Vugt & Luc Nies
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def h(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

def surface_plot():
    """Surface plot of h"""
    X = np.linspace(-2, 2, 100)
    Y = np.linspace(-1, 3, 100)
    [x, y] = np.meshgrid(X, Y)
    z = h(x, y)

    plt.style.use('classic')
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.cm.viridis, linewidth=0, antialiased=False)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$h(x, y)$')
    plt.show()

def gradient_descent(eta=0.005, start_x=0, start_y=0, max_iter=500):
    """
    Do gradient descent on h. Stops when h stops decreasing.
    Note that the minimum of h is at (1, 1)

    parameters:
    eta is the learning rate
    start_x and start_y represent the starting point
    max_iter is the maximum number of iterations before stopping

    returns:
    List of all intermediary x, y and h(x, y)
    """
    x_list, y_list, h_list  = [start_x], [start_y], [h(start_x, start_y)]
    x = start_x
    y = start_y
    for i in range(max_iter):
        x_temp = x - eta * (400 * x * (x**2-y) - 2 + 2 * x)
        y = y - eta * (200 * (y - x**2))
        x = x_temp
        h_new = h(x,y)
        if h_new >= h_list[-1]:
            # Stop when h hasn't decreased in the last step
            break
        x_list.append(x)
        y_list.append(y)
        h_list.append(h(x,y))

    return x_list, y_list, h_list

def contour_trajectory_plot():
    """
    Plot the contour of h and the trajectory
    taken with gradient descent for different values of eta
    """
    X = np.linspace(-2, 2, 100)
    Y = np.linspace(-1, 3, 100)
    [x, y] = np.meshgrid(X, Y)
    z = h(x, y)
    etas = [0.0001, 0.001, 0.005, 0.01]
    levels = np.linspace(z.min(), z.max(), 100)

    for eta in etas:
        x_list, y_list, h_list = gradient_descent(start_x=0, start_y=0, eta=eta, max_iter=1000000)
        plt.plot(x_list, y_list, c='r', label=str(eta), linewidth=2.0)
        plt.contourf(x,y,z, cmap=plt.cm.viridis, levels=levels)
        plt.title('$\eta = {}$ n_steps = {}'.format(eta, len(h_list)))
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar(label='$h$')
        plt.savefig('eta_{}.png'.format(str(eta).replace('.', '_')))
        plt.show()


if __name__ == '__main__':
    contour_trajectory_plot()
