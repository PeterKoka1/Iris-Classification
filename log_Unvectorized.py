import pandas as pd
import numpy as np
import math

def sigmoid(z):
    z = -1 * z
    g_z = float(1 / float((1 + math.exp(z))))

    return g_z

def hypothesis(x, y, theta):
    z = 0

    for i in range(len(theta)):
        theta_x = theta[i] * x[i]
        z += theta_x
    h_x_i = sigmoid(z)

    return h_x_i

def cost_J(X, y, theta, j):
    sse = 0
    m = len(y)
    c = 1 / m

    for i in range(m):
        y_i = y[i]
        x_i = X[i]
        h_x_i = hypothesis(x_i, y_i, theta)

        if y_i == 1:
            h_x_i = -1 * math.log(h_x_i)
        elif y_i == 0:
            h_x_i = -1 * math.log(1 - h_x_i)

        cost = 1 / 2 * (h_x_i - y_i) * x_i[j]
        sse += cost

    return c * sse

def gradientDescent(X, y, theta, m):
    alpha = 0.01
    updated_theta = [t for t in theta]

    for j in range(len(theta)):
        cost = cost_J(X, y, theta, j)
        theta_j = updated_theta[j] - (alpha/m) * cost
        updated_theta[j] = theta_j

    return updated_theta

def log_reg(X, y, theta, iterations):
    m = len(y)

    for i in range(iterations):
        theta_j = gradientDescent(X, y, theta, m)
        theta = theta_j

    return theta