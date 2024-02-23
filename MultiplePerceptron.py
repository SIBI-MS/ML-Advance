import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        return self.sigmoid(self.output)

# Create the first model
mlp1 = MultiLayerPerceptron(2, 4, 2)

# Train the first model (You can replace X1 and y1 with your own data)
X1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y1 = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
mlp1.forward(X1)  # Perform forward pass to get the output of the first model

# Use the output of the first model as input for the second model
output_of_first_model = mlp1.output

# Create the second model with input size equal to the output size of the first model
mlp2 = MultiLayerPerceptron(2, 4, 1)

# Train the second model with the output of the first model as input
# (You can replace X2 and y2 with your own data)
X2 = output_of_first_model
y2 = np.array([[0], [1], [1], [0]])
mlp2.forward(X2)  # Perform forward pass to get the output of the second model
