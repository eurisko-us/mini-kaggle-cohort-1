import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from colby_model import evaluate

print('\nStarting...')

data = pd.read_csv('competition-1/practice_dataset.csv').sort_values(by=['x'])

print('\nFitting...')

colby_rss, colby_predictions = evaluate(data)

print('\nFinished Fitting!')

print('\nColby RSS:', colby_rss)

x = np.array([[x] for x in data['x']])
y = np.array([[y] for y in data['y']])

plt.scatter(x, y, color = "black")
plt.plot(x, colby_predictions, color = "blue")
plt.savefig('competition-1/colby_practice.png')