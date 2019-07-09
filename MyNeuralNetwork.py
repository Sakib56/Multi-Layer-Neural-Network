import numpy as np
import matplotlib.pyplot as plt

class MyNeuralNetwork():

    def __init__(self, layers):
        self.weightShapes =  [(r,c) for r,c in zip(layers[1:], layers[:-1])]
        self.weightMatrices = [np.random.standard_normal(ws) for ws in self.weightShapes]
        self.biasVectors = [np.random.standard_normal((ls, 1)) for ls in layers[1:]]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def diffsigmoid(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def predict(self, x):
        for weights, biases in zip(self.weightMatrices, self.biasVectors):
            x = self.sigmoid(np.matmul(weights, x) + biases)
        return x

    def loadTrainingData(self, path):
        self.trainingData = np.load(path)
        return self.trainingData

    def train(self, learningRate):
        inputMat = self.loadTrainingData('XORdata.npz')['x']
        targetMat = self.loadTrainingData('XORdata.npz')['y']

        for i in range(len(inputMat)):
            inputs = np.reshape(inputMat[i], (2,1))
            outputs = self.predict(inputs)
            targets = targetMat[i]

            outputErrors = targets - outputs

            gradients = outputs * (1 - outputs)
            gradients = np.matmul(gradients, outputErrors)
            gradients = np.matmul(gradients, learningRate)        

        return None




# NN TEST
laySize = (2,4,1)
net = MyNeuralNetwork(laySize)

inputMat = net.loadTrainingData('XORdata.npz')['x']

z = np.reshape(inputMat[0], (2,1))
preds = net.predict(z)
print(z, "\n")
print(preds)

net.train(0.01)