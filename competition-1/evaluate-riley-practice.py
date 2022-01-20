import pandas as pd
import math
import matplotlib.pyplot as plt
from riley_model import logistic_fit

data = pd.read_csv("competition-1/practice_dataset.csv")
data.columns = ['i','x','y']


  
def RSS(data):
  total = 0
  for point in data:
    plt.scatter(point['x'],logistic_fit(point['x']), c = "red")
    plt.scatter(point['x'],point['y'], c = "black")
    total += (logistic_fit(point['x']) - point['y'])**2
  
  return total

plt.savefig("evaluate_riley_practice.png")

print("RSS:"+str(RSS(data.iloc)))
