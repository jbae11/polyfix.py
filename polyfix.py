import numpy as np
import pdb
import scipy.linalg as sl

# by Jin Whan Bae

def polyfix(x, y, n, xfix, yfix, xder='', dydx=''):
    """ Fits polynomial p with degree n to data,
        but specify value at specific points
    """

    nfit = len(x)
    if len(y) != nfit:
        raise ValueError('x and y must have the same size')
    nfix = len(xfix)
    if len(yfix) != nfix:
        raise ValueError('xfit adn yfit must have the same size')
    x = np.vstack(x)
    y = np.vstack(y)
    xfix = np.vstack(xfix)
    yfix = np.vstack(yfix)

    # if derivatives are specified:
    if xder != '':
        try: 
            len(xder)
            xder = xder
            dydx = dydx
        except:
            xder = np.array([xder])
            dydx = np.array([dydx])

    else:
        xder = []
        dydx = []

    nder = len(xder)
    if len(dydx) != nder:
        raise ValueError('xder and dydx must have same size')

    nspec = nfix + nder

    specval = np.vstack((yfix, dydx))
    # first find A and pc such that A*pc = specval
    A = np.zeros((nspec, n+1))
    # specified y values
    for i in range(n+1):
        A[:nfix, i] = np.hstack(np.ones((nfix, 1)) * xfix**(n+1-(i+1)))
    if nder > 0:
        for i in range(n):
            A[nfix:nder+nfix, i] = ((n-(i+1)+1) * np.ones((nder, 1)) * xder**(n-(i+1))).flatten()
    if nfix > 0:
        lastcol = n+1
        nmin = nspec - 1
    else:
        lastcol = n
        nmin = nspec

    if n < nmin:
        raise ValueError('Polynomial degree too low, cannot match all constraints')
    # find unique polynomial of degree nmin that fits the constraints
    firstcol = n-nmin#+1
    pc0 = np.linalg.solve(A[:, firstcol:lastcol], specval)
    pc = np.zeros((n+1, 1))
    pc[firstcol:lastcol] = pc0

    X = np.zeros((nfit, n+1))
    for i in range(n+1):
        X[:, i] = (np.ones((nfit, 1)) * x**(n+1-(i+1))).flatten()

    yfit = y - np.polyval(pc, x)

    B = sl.null_space(A)
    z = np.linalg.lstsq(X @ B, yfit)[0]
    if len(z) == 0:
        z = z[0]
        p0 = B*z
    else:
        p0 = B@z
    p = np.transpose(p0) + np.transpose(pc)
    return p
