from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from colby_model import evaluate

print('\nStarting...')

training_data = pd.read_csv('competition-1/practice_dataset.csv').sort_values(by=['x'])

x_train = np.array([[x] for x in training_data['x']])
y_train = np.array([[y] for y in training_data['y']])

plt.scatter(x_train, y_train, color = "black")

print('\n\tTraining...')

LR = LinearRegression().fit(x_train, y_train)

print('\n\tFinished Training!')

rss = sum([(y - prediction) ** 2 for y, prediction in zip(y_train, LR.predict(x_train))])[0]

plt.plot(x_train, LR.predict(x_train), color = "green")
plt.savefig('competition-1/linear_practice.png')

print('\n\tLinear Train RSS:', rss)

print('\nSaved Practice Plot!')

plt.clf()

print('\n\tTesting...')

testing_data = pd.read_csv('competition-1/competition_dataset.csv').sort_values(by=['x'])

x_test = np.array([[x] for x in testing_data['x']])
y_test = np.array([[y] for y in testing_data['y']])

plt.scatter(x_test, y_test, color = "black")

rss = sum([(y - prediction) ** 2 for y, prediction in zip(y_test, LR.predict(x_test))])[0]

print('\n\tFinished Testing!')

print('\n\tLinear Test RSS:', rss)

plt.plot(x_test, LR.predict(x_test), color = "green")
plt.savefig('competition-1/linear_competion.png')

print('\nSaved Testing Plot!')

print('\nFinished!\n')