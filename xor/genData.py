import numpy as np

def generate():
    oo = [0,0]
    ol = [0,1]
    lo = [1,0]
    ll = [1,1]
    inputsMat = np.matrix([oo,ol,lo,ll])

    o = [0]
    l = [1]
    outputsMat = np.matrix([o,l,l,o])

    # TODO: Fix where .npz saves
    trainingFileStr = 'XORdata.npz'
    np.savez(trainingFileStr, x=inputsMat, y=outputsMat)

    data = np.load(trainingFileStr)
    print("inputs:\n{0}\n\ntargets:\n{1}".format(data['x'], data['y']))