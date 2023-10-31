import numpy as np

def sigmoid(s):
    return 1/(1+np.exp(-s))
    
def softmax(ss):
    m=np.max(ss)
    a=np.exp(ss-m)
    s=np.sum(a)
    return a/s

def _numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) 
    for idx in range(x.size):
        # save x[idx]
        tmp = x[idx]

        # for f(x + h)
        x[idx] = tmp + h
        fh1 = f(x)

        # for f(x - h)
        x[idx] = tmp - h
        fh2 = f(x)

        grad[idx] = (fh1 - fh2)/(2*h)
        # restore x[idx]
        x[idx] = tmp
    return grad

# works with mini-batch as well
def nnumerical_gradient(f, x):
    if x.ndim == 1:
        return _numerical_gradient(f, x)
    else:
        grad = np.zeros_like(x)
        for idx, x in enumerate(x):
            grad[idx] = _numerical_gradient(f, x)
        return grad
    
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0] 
    return -np.sum(t * np.log(y + 1e-7))/batch_size