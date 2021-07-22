import numpy as np
import scipy as sp
import sympy
from sympy.abc import x
from sympy import log
import matplotlib.pyplot as plt


def sigmoid(z):
    """
    Evaluates the sigmoid activation of input z
    :param z: Input weighted sum
    :type z: Numeric (scalar or array)
    :return: Sigmoid activation of z
    """
    return 1 / (1 + np.exp(-1 * z))


def loss_piecewise(y, a):
    """
    Evaluates loss using two conditional expressions
    :param y: Training output
    :type y: scalar or array
    :param a: Predicted output
    :type a: scalar or array
    :return: Loss function value
    """
    if not y:  # Runs if y == 0
        return -1 * np.log10(1 - a)
    else:  # Runs if y != 0
        return -1 * np.log10(a)


def loss_compact(y, a):
    """
    Evaluates loss using single expression
    :param y: Training output
    :type y: scalar or array
    :param a: Predicted output
    :type a: scalar or array
    :return: Loss function value
    """
    return -1 * (y * np.log10(a) + (1 - y) * np.log10(1 - a))


def graph_performance(y, a):  # Add parameters as necessary
    """
    TODO: Graph performance of model, i.e. loss, bias, variance, ...
    :param y: Training output
    :param a: Predicted output
    :return:
    """

# TODO: Rework this entire function, it's quite garbage
def synthesize_all(m = 250, n_x = 5):
    """
    Synthesize all the data - inputs, weights, biases, and outputs
    :param m: Number of examples to synthesize
    :param n_x: Number of features to synthesize
    :return:
    """
    x = synthesize_data_X(n_x, m)
    y = synthesize_data_Y(m)
    w, b = synthesizeDataParams(n_x) # TODO: Readjust dimensions of W here and below :-)
    return x, y, w, b


def synthesize_data_X(n_x, m, mu_x = sp.random.random(1) * 25, sigma_x = sp.random.random(1) * 25):
    """
    Synthesize input matrix X
    :param n_x: Number of features
    :param m: Number of input examples per feature
    :param mu_x: Mean of normal dist. X is sampled from
    :param sigma_x: st. dev of normal dist. X is sampled from
    :return X: Input matrix X
    """
    # Create a (1*m) array that represents m examples of a feature x_i.
    x_i = sp.random.normal(mu_x, sigma_x, m).reshape(1, m)
    # Compile all n_x x_i's into a matrix X, dim (n_x * m).
    X = np.matrix(np.array([sp.random.normal(mu_x, sigma_x, m).reshape(1, m)
                            for i in range(n_x)])).reshape(n_x, m)
    return X, n_x, m, mu_x, sigma_x

# TODO: Make this function able to generate noise around any provided math f'xn we want to model
# i.e. make f a parameter, where f = x**2 + 2*x - 14 + 1 / (1 + exp(-x)), or f = x**3, or whatever :)
# Resource: https://stackoverflow.com/questions/46321333/lambdify-expressions-with-native-sympy-functions
def synthesize_data_Y(m, f = sympy.lambdify(x, log(x), 'numpy')):
    mu_y = sp.random.random(1) * 25
    sigma_y = sp.random.random(1) * 25
    x_i = np.absolute(sp.random.normal(mu_y, sigma_y, m).reshape(1, m))
    y = f(x_i)
    if np.any(x_i <= 0):
        print('there\'s an element in x_i that\'s non-positive')
    return y


def synthesizeDataParams(size, mu_W=0, sigma_W=sp.random.random(1)):
    '''
    Description:
        Synthesizes random data parameters
        drawn from a normal distribution.
    Parameters:
        size: size of W vector - integer
    Optional Parameters:
        mu_W (default = 1): mean of normal
        dist. W is drawn from - integer.
        sigma_W (default ∈ [0, 1)): variance
        of normal dist. W is drawn from - integer.
    Returns:
        W: 'size'-dimensional vector - ndarray.
        b: scalar - ndarray.
        mu_W
        sigma_W
    '''

    W = sp.random.normal(mu_W, sigma_W, size)
    b = sp.random.random(1) * 5

    return W, b, (mu_W, sigma_W)

'''
# Testing noise creation:
n = 100
x = np.linspace(-4, 4, n)
y = np.exp(-1*x**2 / 2)

k = 4
fig, ax = plt.subplots(k//2, k - k//2)

inc = .04
for i in range(k):
    y += i*inc*sp.random.normal(0, 1, n)
    ax[i//2][(i - 1)//2].scatter(x, y)
    ax[i//2][(i - 1)//2].text(0.5, 0.5, i*inc,
                              fontsize=18, ha='center')

plt.show()
'''
m = 100
n_x = 6
x = synthesize_data_X(n_x, m)
y = synthesize_data_Y(m)

if n_x % 2 == 0:
    n_rows = n_x // 2
    n_cols = n_x // (n_x // 2)
else:
    n_rows = n_x // 2
    n_cols = n_x // (n_x // 2) + 1

fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')

c1, c2 = (0, 0)
for i in range(2):
    for j in range(3):
        ax[i][j].scatter(np.array(x[0][i + j]), y)
        ax[i][j].text(0.5, 0.5, c1 + c2,
                      fontsize=18, ha='center')
        c2 += 1
    c1 += 1

# plt.scatter(np.array(x[0][0]), y)
plt.show()
