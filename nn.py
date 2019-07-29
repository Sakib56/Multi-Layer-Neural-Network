import numpy as np

class NeuralNetwork():
    def __init__(self, layers):
        self.weightShapes = [(r, c) for r, c in zip(layers[1:], layers[:-1])]
        self.weightMatrices = [np.random.standard_normal(ws)/ws[1]**0.5 for ws in self.weightShapes]
        self.biasVectors = [np.zeros((ls, 1)) for ls in layers[1:]]

    # prints current weights and biases
    def info(self):
        print("weightMatrices ", self.weightMatrices)
        print("biasVectors ", self.biasVectors)

    # activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # differentiated activation function 
    def diffsigmoid(self, x):
        # no need to run through sigmoid again since x has already been through activation
        return x * (1 - x)

    # feedforward algorithm
    def predict(self, x):
        for weights, biases in zip(self.weightMatrices, self.biasVectors):
            x = self.sigmoid(weights @ x + biases)
        return x

    # loads .npz containing training/testing data
    def loadTrainingData(self, path):
        self.trainingData = np.load(path)
        return self.trainingData

    def feedforward(self, layer, layerIndex):
        return self.sigmoid(self.weightMatrices[layerIndex] @ layer + self.biasVectors[layerIndex])

    # backpropagation algorithm 
    def train(self, learningRate, r):
        inputs = np.reshape(self.trainingData['x'][r], (2, 1))
        targets = np.reshape(self.trainingData['y'][r], (1, 1))

        noOfLayers = len(self.weightShapes)
        feedList = [inputs]
        for i in range(noOfLayers):
            previousLayer = feedList[i]
            feedList.append(self.feedforward(previousLayer, i))

        errorList = [targets - feedList[noOfLayers]]
        errIdx = 0
        for j in range(noOfLayers-1, -1, -1):
            error = errorList[errIdx]
            gradient = self.diffsigmoid(feedList[j+1])
            gradient = (gradient * error) * learningRate
            deltaWeight = gradient @ np.transpose(feedList[j])
            self.weightMatrices[j] += deltaWeight
            self.biasVectors[j] += gradient

            errorList.append(np.transpose(self.weightMatrices[j]) @ error)
            errIdx += 1
        
        # outputErrors = targets - feedList[2]

        # gradients = self.diffsigmoid(feedList[2])
        # gradients = (gradients * outputErrors) * learningRate
        # deltaWeightsHO = gradients @ np.transpose(feedList[1])
        # self.weightMatrices[1] += deltaWeightsHO
        # self.biasVectors[1] += gradients

        # hiddenErrors = np.transpose(self.weightMatrices[1]) @ outputErrors
        # hiddenGradients = self.diffsigmoid(feedList[1])
        # hiddenGradients = (hiddenGradients * hiddenErrors) * learningRate
        # deltaWeightsIH = hiddenGradients @ np.transpose(feedList[0])

        # self.weightMatrices[0] += deltaWeightsIH
        # self.biasVectors[0] += hiddenGradients

## MAIN ###
# layers = (2, 4, 3, 1)
# net = NeuralNetwork(layers)
# data = net.loadTrainingData('XORdata.npz')

# import random
# net.train(0.5, random.choice([0, 1, 2, 3]))
