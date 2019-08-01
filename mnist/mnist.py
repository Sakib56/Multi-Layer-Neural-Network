import sys
sys.path.append(sys.path[0][:-6])

import nn
import numpy as np

class mnist():
    def __init__(self, neuralNetwork):
        self.net = neuralNetwork
        self.avgError = 0
        self.noOfTests = 0

    def teachModel(self, learningRate, trainingTime, noOfDataPoints):
        print("training...\n")
        import random
        import time

        self.iterationsSoFar = 0
        timeLimit = time.time() + trainingTime
        while time.time() < timeLimit:
            self.net.train(learningRate, random.choice([i for i in range(noOfDataPoints)]))
            self.iterationsSoFar += 1
        print("training completed in {0:.2f}s\n".format(time.time() - timeLimit + trainingTime))

    def testModel(self, testImages, testLabels):
        print("post-training tests")
        correct = 0
        total = testLabels.shape[0]
        for img in range(total):
            predication = np.argmax(net.predict(testImages[img]))
            observed = np.argmax(testLabels[img])
            if predication == observed:
                correct += 1
        print("lr: {0:.5f}     totalIter: {1}     acc: {2:.5f}%".format(learningRate, self.iterationsSoFar, 100*correct/total))


## MAIN ###
# TODO: add some sort of gui/cli for using nn
layers = (784, 16, 10)
learningRate = 0.1
trainingTime = 15
data = np.load('MNISTdata.npz')
noOfDataPoints = data['training_images'].shape[0]

net = nn.NeuralNetwork(layers)
net.loadData('training_images', 'training_labels', data)

mnsitModel = mnist(net)
mnsitModel.teachModel(learningRate, trainingTime, noOfDataPoints)
mnsitModel.testModel(data['test_images'], data['test_labels'])
