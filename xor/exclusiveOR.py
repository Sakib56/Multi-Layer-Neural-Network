import sys 
sys.path.append(sys.path[0][:-4])

import nn
import numpy as np

class xor():
    def __init__(self, neuralNetwork):
        self.net = neuralNetwork
        self.avgError = 0
        self.noOfTests = 0

    def generateData(self):
        import genData as xorData
        xorData.generate()

    def teachModel(self, learningRate, trainingIter, noOfDataPoints):
        import random
        print("\ntraining...\n")
        for j in range(trainingIter):
            self.net.train(learningRate, random.choice([i for i in range(noOfDataPoints)]))
        print("training complete!\n")

    def testModel(self):
        print("post-training tests")#
        self.noOfTests += 1
        for i in range(4):
            x = np.reshape(data['x'][i], (2, 1))
            y = np.reshape(data['y'][i], (1, 1))
            predications = self.net.predict(x)
            error = abs(y-self.net.predict(x))[0][0]
            self.avgError += error
            print("input:\n{0}\ntarget:{1}\nprediction:{2}\nerror:{3:.3f}\n".format(x, y, predications, error, self.avgError))
        print("avg error: {0:.3f}".format(self.avgError/self.noOfTests))


## MAIN ###
layers = (2, 8, 6, 8, 1)
net = nn.NeuralNetwork(layers)

xorModel = xor(net)

xorModel.generateData()
data = np.load('XORdata.npz')
net.loadData('x', 'y', data)

noOfDataPoints = data['x'].shape[0]
learningRate = 0.05
trainingIter = 10000

xorModel.teachModel(learningRate, trainingIter, noOfDataPoints)
xorModel.testModel()