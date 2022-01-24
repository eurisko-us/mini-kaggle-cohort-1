import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from logistic_model import evaluate

print('\nStarting...')

training_data = pd.read_csv('competition-1/practice_dataset.csv').sort_values(by=['x'])

x_train = np.array([[x] for x in training_data['x']])
y_train = np.array([[y] for y in training_data['y']])

plt.scatter(x_train, y_train, color = "black")

print('\n\tTraining...')

precision = 0.0001

coeffs = evaluate(training_data)

m = coeffs[0].value
M = coeffs[1].value
a = coeffs[2].value
b = coeffs[3].value

predictions = [float(m + (M - m) / (1 + math.e ** (a * x + b))) for x in x_train]
rss = sum([(y - prediction) ** 2 for y, prediction in zip(y_train, predictions)])[0]

print('\n\tFinished Training!')

print('\n\tLogistic Train RSS:', rss)

plt.plot(x_train, predictions, color = "blue")
plt.savefig('competition-1/logistic_practice.png')

print('\nSaved Practice Plot!')

plt.clf()

print('\n\tTesting...')

testing_data = pd.read_csv('competition-1/competition_dataset.csv').sort_values(by=['x'])

x_test = np.array([[x] for x in testing_data['x']])
y_test = np.array([[y] for y in testing_data['y']])

plt.scatter(x_test, y_test, color = "black")

predictions = [float(m + (M - m) / (1 + math.e ** (a * x + b))) for x in x_test]
rss = sum([(y - prediction) ** 2 for y, prediction in zip(y_test, predictions)])[0]

print('\n\tFinished Testing!')

print('\n\tLogistic Test RSS:', rss)

plt.plot(x_test, predictions, color = "blue")
plt.savefig('competition-1/logistic_competion.png')

print('\nSaved Testing Plot!')

print('\nFinished!\n')