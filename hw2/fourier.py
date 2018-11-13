def squaredEDM(X):
    V = spt.distance.pdist(X, 'sqeuclidean')
    D = spt.distance.squareform(V)
    return D

def gaussianKernelMat(X, sigma):
    D = squaredEDM(X)
    K = np.exp(-0.5 / sigma**2 * D)
    return np.fft.fft(K)

def gaussianKernelVec(x, X, sigma):
    d = np.sum((X-x)**2. , axis = 1)
    k = np.exp(-0.5/sigma**2 * d)
    return np.fft.fft(k)

def gaussian_lss(X, y, sigma = 2.5):
    n = X.shape[0]
    K = gaussianKernelMat(X, sigma)
    KI = la.inv(K + 1. * np.identity(n))
    KIy = np.dot(KI, y)
    return KIy
