import numpy as np

from layer import Layer

class Network:
    def __init__(self, dims: list[int]):
        self.layers = []
        for i in range(len(dims)-1):
            self.layers.append(Layer(dims[i], dims[i+1]))

    def forward(self, input_vals):
        layer_outputs = []
        for layer in self.layers:
            input_vals = layer.forward(input_vals)
            layer_outputs.append(input_vals)
        return layer_outputs
    
    def backward(self, input_vals, layer_outputs, label_one_hot):
        d_weights = []
        d_biases = []

        # Last layer done on its own to use the label
        d_layer = layer_outputs[-1] - label_one_hot
        d_weights.append(d_layer @ layer_outputs[-2].T)
        d_biases.append(d_layer)

        for i in range(len(layer_outputs)-1, 0, -1):
            curr_layer = self.layers[i]

            d_layer = curr_layer.weights.T @ d_layer

            prev_input = layer_outputs[i-1] if i > 1 else input_vals.reshape(-1, 1)
            d_weights.append(d_layer @ prev_input.T)
            d_biases.append(d_layer)    

        d_weights.reverse()
        d_biases.reverse()

        return d_weights, d_biases
    
    def update(self, d_weights, d_biases, alpha):
        for layer, d_weight, d_bias in zip(self.layers, d_weights, d_biases):
            layer.weights -= alpha * d_weight
            layer.biases -= alpha * d_bias
    

if __name__ == "__main__":
    net = Network([4,5,2])
    input_vals = np.array([0.5,0.1,0.8,0.6]).reshape(-1, 1)
    layer_outputs = net.forward(input_vals)
    d_weights, d_biases = net.backward(input_vals, layer_outputs, np.array([0,1]).reshape(-1, 1))
    net.update(d_weights, d_biases, 0.1)
    print(net.layers[-1].biases)