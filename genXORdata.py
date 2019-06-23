import numpy as np

inputsMat = np.mat([[0,0],[0,1],[1,0],[1,1]])
outputsMat = np.mat([[0],[1],[1],[0]])

trainingFileStr = 'XORdata.npz'
np.savez(trainingFileStr, x=inputsMat, y=outputsMat)

trainingData = np.load(trainingFileStr)
print(trainingData.files, "\n")
print(trainingData['x'], "\n")
print(trainingData['y'])