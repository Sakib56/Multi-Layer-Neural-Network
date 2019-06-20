import numpy as np
import matplotlib.pyplot as plt

class MyNeuralNetwork():

    def __init__(self, layers):
        self.weightShapes =  [(r,c) for r,c in zip(layers[1:], layers[:-1])]
        self.weightMatrices = [np.random.standard_normal(ws) for ws in self.weightShapes]
        self.biasVectors = [np.random.standard_normal((ls, 1)) for ls in layers[1:]]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        for weights, biases in zip(self.weightMatrices, self.biasVectors):
            x = self.sigmoid(np.matmul(weights, x) + biases)
        return x

    def loadTrainingData(self, path):
        return None

    def train(self, amount, learningRate):
        return None
    
    def visualise(self):
        return None

# NN TEST
laySize = (3,5,10)
x = np.ones((laySize[0],1))
net = MyNeuralNetwork(laySize)
preds = net.predict(x)
print(preds)