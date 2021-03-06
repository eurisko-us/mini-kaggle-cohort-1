import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def evaluate(data):

    precision = 0.0001

    x = np.array([[x] for x in data['x']])
    y = np.array([[y] for y in data['y']])

    max_y = max(y) + precision
    min_y = min(y) - precision

    x_transformed = np.array([[x[0], 1] for x in x])
    y_transformed = np.array([[math.log((max_y - y)/(y - min_y))] for y in y])

    return ((np.linalg.inv(x_transformed.T @ x_transformed)) @ x_transformed.T) @ y_transformed