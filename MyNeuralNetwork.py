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
        self.trainingData = np.load(path)
        return self.trainingData

    def train(self, amount, learningRate):
        return None
    
    def visualise(self):
        return None

# NN TEST
laySize = (2,4,1)
net = MyNeuralNetwork(laySize)

inputMat = net.loadTrainingData('XORdata.npz')['x']
z = np.reshape(inputMat[0], (2,1))

preds = net.predict(z)

print(z, "\n")
print(preds)