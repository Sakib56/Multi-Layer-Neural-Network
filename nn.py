import numpy as np

class NeuralNetwork():
    def __init__(self, layers):
        self.weightShapes = [(r, c) for r, c in zip(layers[1:], layers[:-1])]
        self.weightMatrices = [np.random.standard_normal(ws)/ws[1]**0.5 for ws in self.weightShapes]
        self.biasVectors = [np.zeros((ls, 1)) for ls in layers[1:]]

    # prints current weights and biases
    def info(self):
        layerNumber = 0
        for weights, biases in zip(self.weightMatrices, self.biasVectors):
            print("\nweight{2}: \n{0}\nbias{2}: \n{1}\n".format(weights, biases, layerNumber))
            layerNumber += 1

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
    def loadData(self, inputsIdx, labelsIdx, data):
        self.inputs = data[inputsIdx]
        self.labels = data[labelsIdx]

    def feedforward(self, layer, layerIndex):
        return self.sigmoid(self.weightMatrices[layerIndex] @ layer + self.biasVectors[layerIndex])

    # backpropagation algorithm
    def train(self, learningRate, rowInData):
        inputs = np.reshape(self.inputs[rowInData], (self.inputs[rowInData].shape[0], 1))
        targets = np.reshape(self.labels[rowInData], (self.labels[rowInData].shape[0], 1))

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
