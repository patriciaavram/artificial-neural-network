import numpy as np

class Layer:
    def __init__(self, input_dim: int, output_dim: int):
        self.weights = np.zeros([output_dim, input_dim])
        self.biases = np.zeros([output_dim, 1])

    def forward(self, input_vals: np.ndarray) -> np.ndarray:
        return self.weights @ input_vals + self.biases
    

if __name__ == "__main__":
    layer = Layer(2,3)
    print(layer.forward(np.array([0.1, 0.9]).reshape(-1, 1)))