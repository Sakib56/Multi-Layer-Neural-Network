import sys 
sys.path.append(sys.path[0][:-4])
import nn

def getData():
    import genXORdata
    genXORdata.generate()

def teachModel():
    import random
    print("\ntraining...\n")
    for j in range(trainingIter):
        net.train(learningRate, random.choice([0, 1, 2, 3]))
    print("training complete!\n")

def testModel():
    import numpy as np
    print("post-training tests")
    totalErr = 0
    for i in range(4):
        x = np.reshape(data['x'][i], (2, 1))
        y = np.reshape(data['y'][i], (1, 1))
        preds = net.predict(x)
        err = abs(y-net.predict(x))[0][0]
        totalErr += err
        print("input:\n{0}\ntarget:{1}\nprediction:{2}\nerror:{3:.3f}\n".format(x, y, preds, err, totalErr))
    print("total error: {0:.3f}".format(totalErr))


## MAIN ###
layers = (2, 8, 6, 8, 1)
net = nn.NeuralNetwork(layers)

getData()
data = net.loadTrainingData('XORdata.npz')

learningRate = 0.05
trainingIter = 10000
teachModel()
testModel()