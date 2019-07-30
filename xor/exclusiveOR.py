import sys 
sys.path.append(sys.path[0][:-4])

import genXORdata
import nn
import random
import numpy as np

def generateData():
    genXORdata.generate()

def teachModel():
    print("\ntraining...\n")
    for j in range(trainingIter):
        net.train(learningRate, random.choice([0, 1, 2, 3]))
    print("training complete!\n")

def testModel():
    print("post-training tests")
    totalError = 0
    for i in range(4):
        x = np.reshape(data['x'][i], (2, 1))
        y = np.reshape(data['y'][i], (1, 1))
        predications = net.predict(x)
        error = abs(y-net.predict(x))[0][0]
        totalError += error
        print("input:\n{0}\ntarget:{1}\nprediction:{2}\nerroror:{3:.3f}\n".format(x, y, predications, error, totalError))
    print("total erroror: {0:.3f}".format(totalError))


## MAIN ###
layers = (2, 8, 6, 8, 1)
net = nn.NeuralNetwork(layers)

generateData()

data = np.load('XORdata.npz')
net.loadData('x', 'y', data)

learningRate = 0.05
trainingIter = 10000
teachModel()
testModel()