import numpy as np

def bin2Dec (x,n,m):
    y = x.reshape(n,m)
    w = np.matlib.repmat(2** (np.arange(m-1,-1,-1)),n,1)
    z = np.sum(y*w,axis = 1)
    return (z)


def dec2Bin (z,n,m):
    result = np.zeros((n,m),int)
    for i in range(n):
        a = list(bin(z[i]))
        b = a[2:]
        while len(b)<m:
            b.insert(0, '0')        
        for j in range(m):
            result[i,j] = int(b[j])
    result2 = result.reshape(n*m,1)
    return (result2)
