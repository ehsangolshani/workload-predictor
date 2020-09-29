class ARMAModel():
    def __init__(self, gamma=0.15, beta=0.8):
        self.gamma_param = gamma
        self.beta_param = beta

    def predict(self, x):
        x_simplified = x[0, 0, :]  # convert to a simple array, ignoring batch size
        return (self.beta_param * x_simplified[-1]) + \
               (self.gamma_param * x_simplified[-2]) + \
               ((1 - (self.gamma_param + self.beta_param)) * x_simplified[-3])
