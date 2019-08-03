import numpy as np

class NeuralNetwork():
    def __init__(self, layers):
        self.weightShapes = [(r, c) for r, c in zip(layers[1:], layers[:-1])]
        self.weightMatrices = [np.random.standard_normal(ws)/ws[1]**0.5 for ws in self.weightShapes]
        self.biasVectors = [np.zeros((ls, 1)) for ls in layers[1:]]
        self.setActivation(self, "sigmoid")

    # prints current weights and biases
    def info(self):
        layerNumber = 0
        for weights, biases in zip(self.weightMatrices, self.biasVectors):
            print("\nweight{2}: \n{0}\nbias{2}: \n{1}\n".format(weights, biases, layerNumber))
            layerNumber += 1

    # sets activation func and differentiated activation func, defualt is sigmoid
    def setActivation(self, type):
        if type == "sigmoid":
            self.activation = lambda x : 1 / (1 + np.exp(-x))
            self.diffActivation = lambda x: x * (1 - x)
        if type == "tanh":
            self.activation = lambda x : np.tanh(x)
            self.diffActivation = lambda x: 1 - (x * x)
        if type == "relu":
            self.activation = lambda x : x * (x > 0)
            self.diffActivation = lambda x: (x > 0).astype(int)

    # feedforward algorithm
    def predict(self, x):
        for weights, biases in zip(self.weightMatrices, self.biasVectors):
            x = self.activation(weights @ x + biases)
        return x

    # loads .npz containing training/testing data
    def loadData(self, inputsIdx, labelsIdx, data):
        self.inputs = data[inputsIdx]
        self.labels = data[labelsIdx]

    def feedForward(self, layer, layerIndex):
        return self.activation(self.weightMatrices[layerIndex] @ layer + self.biasVectors[layerIndex])

    # backpropagation algorithm
    def train(self, learningRate, rowInData):
        inputs = np.reshape(self.inputs[rowInData], (self.inputs[rowInData].shape[0], 1))
        targets = np.reshape(self.labels[rowInData], (self.labels[rowInData].shape[0], 1))

        noOfLayers = len(self.weightShapes)
        feedList = [inputs]
        for i in range(noOfLayers):
            previousLayer = feedList[i]
            feedList.append(self.feedForward(previousLayer, i))

        errorList = [targets - feedList[noOfLayers]]
        errIdx = 0
        for j in range(noOfLayers-1, -1, -1):
            error = errorList[errIdx]
            gradient = self.diffActivation(feedList[j+1])
            gradient = (gradient * error) * learningRate
            deltaWeight = gradient @ np.transpose(feedList[j])
            self.weightMatrices[j] += deltaWeight
            self.biasVectors[j] += gradient

            errorList.append(np.transpose(self.weightMatrices[j]) @ error)
            errIdx += 1
