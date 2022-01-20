import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from colby_model import evaluate

print('\nStarting...')

training_data = pd.read_csv('competition-1/practice_dataset.csv').sort_values(by=['x'])

x_train = np.array([[x] for x in training_data['x']])
y_train = np.array([[y] for y in training_data['y']])

plt.scatter(x_train, y_train, color = "black")

print('\nFitting...')

coeffs = evaluate(training_data)

a = 0.875 * coeffs[0][0]
b = 0.875 * coeffs[1][0]

predictions = [float(min_y + (max_y - min_y) / (1 + math.e ** (a * x + b))) for x in x_train]
rss = sum([(y - prediction) ** 2 for y, prediction in zip(y_train, predictions)])[0]

print('\nFinished Training!')

print('\nColby Train RSS:', rss)

plt.plot(x_train, predictions, color = "blue")
plt.savefig('competition-1/colby_practice.png')

print('\nSaved Practice Plot')

plt.clf()

testing_data = pd.read_csv('competition-1/competition_dataset.csv').sort_values(by=['x'])

x_test = np.array([[x] for x in testing_data['x']])
y_test = np.array([[y] for y in testing_data['y']])

plt.scatter(x_test, y_test, color = "black")

predictions = [float(min_y + (max_y - min_y) / (1 + math.e ** (a * x + b))) for x in x_test]
rss = sum([(y - prediction) ** 2 for y, prediction in zip(y_test, predictions)])[0]

print('\nColby Test RSS:', rss)

print('\nFinished Testing!')

plt.plot(x_test, predictions, color = "blue")
plt.savefig('competition-1/colby_competion.png')

print('\nSaved Testing Plot!')
