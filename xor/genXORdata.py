import numpy as np

def generate():
    inputsMat = np.mat([[0,0],[0,1],[1,0],[1,1]])
    outputsMat = np.mat([[0],[1],[1],[0]])

    trainingFileStr = 'XORdata.npz'
    np.savez(trainingFileStr, x=inputsMat, y=outputsMat)

    data = np.load(trainingFileStr)

    print("inputs:\n{0}\n\ntargets:\n{1}".format(data['x'], data['y']))