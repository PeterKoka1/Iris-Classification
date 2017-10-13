import pandas as pd
from numpy import dot, ones, exp, log, subtract, matrix, shape, repeat

def sigmoid(z):
    g = (1 / (1 + exp(-z)))
    return g

def logReg(X, y, theta, m):
    const = [-1 / m, 1 / m]

    lamb = 1
    h_x_i = sigmoid(dot(X, theta))
    J = dot(y.T, log(h_x_i)) + dot(subtract(1, y).T, log(subtract(1, h_x_i)))
    grad = dot(X.T, subtract(h_x_i, y)) + (lamb / m) * theta

    J *= const[0]
    grad *= const[1]

    return grad

def iris():
    data = pd.read_csv('C:\\Users\\PeterKokalov\\lpthw\\Projects\\Iris_Classification\\Iris.csv')
    X = data.drop(labels=['Species','Id'], axis=1)
    X['ones'] = ones(len(data))
    X = X[:-50]
    X = X.as_matrix()
    y = data['Species']
    y = [species for species in y if species != 'Iris-virginica']
    species = repeat([1], len(y))
    for i, j in enumerate(species):
        if y[i] == 'Iris-setosa':
            species[i] = 0
    y = matrix(species).T
    y = y.reshape(len(y), 1)
    theta = matrix('0.5;0.5;0.5;1.4;1.4')
    m = len(X)

    opt_theta = logReg(X, y, theta, m)

    print(opt_theta)

iris()

