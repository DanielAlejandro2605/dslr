import pandas as pd
import numpy as np
# Import plots
from modules.plotting import plot_cost_function_scatter

def derivative_of_w(predictions : np.ndarray, data_x : np.ndarray, data_y : np.ndarray) -> float:
    # Training examples
    m: int = data_x.shape[0]
    # Transporte data x
    data_x_transpose = data_x.transpose()

    data_y = data_y.flatten()
    # Compute deviation
    deviation = np.matmul(data_x_transpose, (predictions - data_y))
    deviation = deviation * (1 / m)
    return deviation

def sigmoid_function(z : np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def compute_cost_ft(data_x : np.ndarray, data_y : np.ndarray, weigths : np.ndarray):
    # Training examples
    data_y = data_y.flatten()
    m = len(data_y)
    # Predictions
    predictions = sigmoid_function(np.dot(data_x, weigths))
    # Cost functions
    cost = - (1 / m) * np.sum(data_y * np.log(predictions) + (1 - data_y) * np.log(1 - predictions))
    return cost

def logistic_regression(data_x: np.ndarray, data_y: np.ndarray, house : str, tolerance: float = 1e-6):
    # Init weigths
    w = np.zeros(data_x.shape[1])

    # save_data_to_file(data_x, data_y, 'data_output.txt')
    # Learning rate
    learning_rate = 0.0001
    # For plot function
    costs = []
    iterations = []

    i = 0
    while True:
        # Performing predictions
        predictions = sigmoid_function(np.dot(data_x, w))
        # Perfoming derivatives for gradient descent decision
        dj_dw = derivative_of_w(predictions, data_x, data_y)
        # New array of weigths
        new_w = w - learning_rate * dj_dw
        # Computing costs for plot
        if i % 1000 == 0:
            cost = compute_cost_ft(data_x, data_y, new_w)
            costs.append(cost)
            iterations.append(i)

        if np.linalg.norm(new_w - w) < tolerance:
            break

        w = new_w
        i += 1

    # plot_cost_function_scatter(iterations, costs, f'./plots/cost_function_scatter_{house}.png')
    return w, i

def predict(data_x: np.ndarray, w: np.ndarray) -> np.ndarray:

    z = np.dot(data_x, w)

    probabilities = sigmoid_function(z)

    return probabilities
