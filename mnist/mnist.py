import sys 
sys.path.append(sys.path[0][:-6])
import nn
import numpy as np

def teachModel():
    import random
    print("\ntraining...\n")
    for j in range(trainingIter):
        net.train(learningRate, random.choice([i for i in range(trainingImages.shape[0])]))
    print("training complete!\n")

def testModel():
    print("post-training tests")
    correct = 0
    total = testLabels.shape[0]
    for img in range(total):
        predication = np.argmax(net.predict(testImages[img]))
        observed = np.argmax(testLabels[img])
        if predication == observed:
            correct += 1
    print("\nacc: {0}%".format(100*correct/total))

## MAIN ###
layers = (784, 16, 16, 10)
net = nn.NeuralNetwork(layers)

data = np.load('MNISTdata.npz')
net.loadData('training_images', 'training_labels', data)

trainingImages = data['training_images']
trainingLabels = data['training_labels']

testImages = data['test_images']
testLabels = data['test_labels']

learningRate = 0.1
trainingIter = 10000
teachModel()
testModel()