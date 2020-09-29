class LinearRegressionModel():

    def predict(self, x):
        x_refined = x[0, 0, :]  # convert to a simple array, ignoring batch size
        n = len(x_refined)
        x_square_sum = 0
        x_sum = 0
        y_sum = 0
        x_y_mul_sum = 0

        for i, value in enumerate(x_refined):
            x_sum += i + 1
            x_square_sum += (i + 1) * (i + 1)
            y_sum += value
            x_y_mul_sum += (i + 1) * value

        # TODO: make sure theyare float and they doesn't round up
        beta1: float = ((x_square_sum * y_sum) - (x_sum * x_y_mul_sum)) / ((n * x_square_sum) - (x_sum * x_sum))
        beta2: float = ((n * x_y_mul_sum) - (x_sum * y_sum)) / ((n * x_square_sum) - (x_sum * x_sum))

        return beta1 + (beta2 * (n + 1))
