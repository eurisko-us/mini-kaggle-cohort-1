import math

class Weight:
    def __init__(self, initial_value, m = 0, m_cap = 0, v = 0, v_cap = 0):
        self.value = initial_value
        self.alpha = 10 ** -2 # Learning Rate
        self.epsilon = 10 ** -8 # Smol number
        self.ddw = 0 # d/dw if that wasn't obvious enough
        # ADAM Variables
        self.m = 0
        self.m_cap = 0
        self.v = 0
        self.v_cap = 0
        self.bias = (0.9, 0.999)

    def update_weight(self, t):
        self.m = self.bias[0] * self.m + (0.1) * self.ddw
        self.v = self.bias[1] * self.v + (0.0001) * (self.ddw ** 2)
        self.m_cap = self.m / (1 - (self.bias[0] ** t))
        self.v_cap = self.v / (1 - (self.bias[1] ** t))
        self.value -= (self.alpha / (math.sqrt(self.v_cap) + self.epsilon)) * self.m_cap

def evaluate(data): # ADAM Descent Optimizer with CDF for each weights d/dw
    m = Weight(-1.0)
    M = Weight(1.0)
    a = Weight(0)
    b = Weight(0)
    weights = [m, M, a, b]

    for t in range(1, 25000):
        weight_values = [weight.value for weight in weights]
        for i, weight in enumerate(weights):
            weight.ddw = CDF(weight_values, i, data['x'], data['y'])
            weight.update_weight(t)
    return weights # m, M, a, b

def CDF(weights, i, x, y):
    delta = 10 ** -4 # CDF
    cdf = 0
    cdf += calc_rss(x, y, *list(weights[:i] + [weights[i] + delta] + weights[i+1:])) / (2 * delta)
    cdf -= calc_rss(x, y, *list(weights[:i] + [weights[i] - delta] + weights[i+1:])) / (2 * delta)
    return cdf

def calc_rss(x_data, y_data, m, M, a, b):
    predictions = [float(m + (M - m) / (1 + math.e ** (a * x + b))) for x in x_data]
    return sum([(y - prediction) ** 2 for y, prediction in zip(y_data, predictions)])