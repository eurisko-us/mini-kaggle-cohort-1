import math

def logistic_fit(x):
  b_0, b_1 = (2.5389802665309413, -0.5565010793760151)
  return 8/(1+math.e**(-1*(b_0 + b_1*x))) - 2
