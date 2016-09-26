"""
Code for assignment 1, exercise 1 of Statistical Machine Learning

Author: Joris van Vugt & Luc Nies
"""
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 1 + np.sin(6 * (x - 2))

def noisy_f(x):
    noise = np.random.normal(0, 0.3)
    return noise + f(x)

def polynomial(X, w):
    """Evaluate a polynomial with weights w at point X"""
    return np.polyval(list(reversed(w)), X)

def RMSE(observed, target):
    """Calculate the root mean squared error"""
    error = 0.5 * np.sum((observed - target)**2)
    return np.sqrt(2*error / len(observed))

def pol_cur_fit(D, M):
    """Fit weights for an Mth order polynomial with data D"""
    x = D[0, :]
    t = D[1, :]
    A = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            A[i, j] = np.sum(x ** (i+j))
    T = np.zeros(M)
    for i in range(M):
        T[i] = np.sum(t * x**i)
    w = np.linalg.solve(A, T)
    return w

def pol_cur_fit_reg(D, M, l=0.01):
    """Fit regularized weights for an Mth order polynomial with data D"""
    x = D[0, :]
    t = D[1, :]
    A = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            A[i, j] = np.sum(x ** (i+j))
    A += l * np.identity(M)
    T = np.zeros(M)
    for i in range(M):
        T[i] = np.sum(t * x**i)
    w = np.linalg.solve(A, T)
    return w

def generate_data(n):
    """Generate n data points"""
    return [noisy_f(x) for x in np.linspace(0, 1, n)]


def plot_data():
    """Plot data for exercise 1.1"""
    D = generate_data(10)
    T = generate_data(100)
    plt.scatter(np.linspace(0, 1, 10), D)
    X = np.linspace(0, 1, 100)
    y = [f(x) for x in X]
    plt.plot(X, y)
    plt.savefig('sin_d.png')
    plt.show()

def plot_curvefitting():
    """Plot fitted curve for different values of M"""
    N = 10
    N_test = 10
    T = generate_data(N_test)
    D = generate_data(N)
    training_data = np.vstack((np.linspace(0, 1, N), D))
    test_data = np.vstack((np.linspace(0, 1, N_test), T))

    X = np.linspace(0, 1, 100)
    y = [f(x) for x in X]
    # Calculate and plot the errors of different m values (for exercise 1.3)
    for m in range(10):
        w = pol_cur_fit(training_data, m)
        fitted_curve = polynomial(X, w)

        rms_train = RMSE(polynomial(training_data[0, :], w), training_data[1, :])
        rms_test = RMSE(polynomial(test_data[0, :], w), test_data[1, :])

        plt.plot(X, fitted_curve, 'b', label='Fitted Curve')
        plt.plot(X, y, 'r', label='True function')
        plt.scatter(training_data[0, :], training_data[1, :], c='g', label='Noisy observations')
        plt.title('M=%d: train RMSE=%.2f, test RMSE=%.2f' % (m, rms_train, rms_test))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.legend()
        plt.savefig('images/fit_m%d_n_%d.png' % (m, training_data.shape[1]))
        plt.show()

def reg_test():
    """Plot the effect of regularization"""
    N = 10
    N_test = 100
    X = np.linspace(0, 1, N_test)
    y = [f(x) for x in X]
    T = generate_data(N_test)
    D = generate_data(N)
    training_data = np.vstack((np.linspace(0, 1, N), D))
    test_data = np.vstack((np.linspace(0, 1, N_test), T))
    l = 0.1
    for m in range(10):
        w = pol_cur_fit_reg(training_data, m, l=l)
        w_noreg = pol_cur_fit(training_data, m)
        print('regularized w:', w)
        print('nonreg w:', w_noreg)
        fitted_curve = polynomial(X, w)

        rms_train = RMSE(polynomial(training_data[0, :], w), training_data[1, :])
        rms_test = RMSE(polynomial(test_data[0, :], w), test_data[1, :])

        plt.plot(X, fitted_curve, 'b', label='Fitted Curve')
        plt.plot(X, y, 'r', label='True function')
        plt.scatter(training_data[0, :], training_data[1, :], c='g', label='Noisy observations')
        plt.title(r'M=%d: train RMSE=%.2f, test RMSE=%.2f, $\lambda=%.1g$' % (m, rms_train, rms_test, l))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.legend()
        plt.savefig('images/fit_m%d_n_%d_reg.png' % (m, training_data.shape[1]))
        plt.show()


def train_vs_test_reg():
    """Tests and plots the difference between the train and testset (with regularization)"""
    ls = np.arange(-40, -20)
    exp_l = np.exp(ls)
    errors_train = []
    errors_test = []

    N = 10
    N_test = 10
    T = generate_data(N_test)
    D = generate_data(N)
    training_data = np.vstack((np.linspace(0, 1, N), D))
    test_data = np.vstack((np.linspace(0, 1, N_test), T))

    for l in exp_l:
        w = pol_cur_fit_reg(training_data, 9, l=l)
        rms_train = RMSE(polynomial(training_data[0, :], w), training_data[1, :])
        rms_test = RMSE(polynomial(test_data[0, :], w), test_data[1, :])
        errors_train.append(rms_train)
        errors_test.append(rms_test)

    plt.plot(ls, errors_test, 'r', label='Test')
    plt.plot(ls, errors_train, 'b', label='Training')
    plt.ylim([0, 0.4])
    plt.legend()
    plt.xlabel(r'$\ln {\lambda}$')
    plt.ylabel(r'$E_{RMS}}$')
    plt.show()

if __name__ == '__main__':
    train_vs_test_reg()
