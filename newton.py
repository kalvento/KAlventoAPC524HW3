'''newton.py

Implementation of a Newton-Raphson root-finder.

'''

import numpy as np
import functions as F

class Newton(object):
    """Newton objects have a solve() method for finding roots of f(x)
    using Newton's method. x and f can both be vector-valued.

    """
    
    def __init__(self, f, tol=1.e-6, maxiter=20, dx=1.e-6, radius_max=None, Df=None):
        """Parameters:
        
        f: the function whose roots we seek. Can be scalar- or
        vector-valued. If the latter, must return an object of the
        same "shape" as its input x

        tol, maxiter: iterate until ||f(x)|| < tol or until you
        perform maxiter iterations, whichever happens first

        dx: step size for computing approximate Jacobian
        
        radius_max: will bind the root to a range around a specified maximum
        
        Df: analytical jacobian to assist in root finding
        """
        self._f = f
        self._tol = tol
        self._maxiter = maxiter
        self._dx = dx
        self.r = radius_max
        
        
        #set analytical Jacobian if provided
        self._given_analytical_jacobian = False
        if Df is not None:
            self._df = Df
            self._given_analytical_jacobian = True
            

    def solve(self, x0):
        """Determine a solution of f(x) = 0, using Newton's method, starting
        from initial guess x0. x0 must be scalar or 1D-array-like. If
        x0 is scalar, return a scalar result; if x0 is 1D, return a 1D
        numpy array.

        """
        # NOTE: no need to check whether x0 is scalar or vector. All
        # the functions/methods invoked inside solve() return "the
        # right thing" when x0 is scalar.
        x = x0
        for i in range(self._maxiter):
            fx = self._f(x)
            # linalg.norm works fine on scalar inputs
            if np.linalg.norm(fx) < self._tol and self._r is None:
                return x
            x = self.step(x, fx)
        #Add maximum radius
        if self.r is not None:
            if abs(x - x0) <= self.r:
                return x
            else:
                raise Exception("The solution is no longer within the maximum radius, try a new guess")
                
        return x

    def step(self, x, fx=None):
        """Take a single step of a Newton method, starting from x. If the
        argument fx is provided, assumes fx = f(x).

        """
        if fx is None:
            fx = self._f(x)
            
        #If analytical Jacobian is provided, proceed this way:
        if self._given_analytical_jacobian:
            Df_x = self.df(x)
            h = np.linalg.solve(np.matrix(Df_x), np.matrix(fx))
            if np.isscalar(x):
                h = np.asscalar(h)
            return x - h
        else:
        #if it is not provided:
            Df_x = F.approximateJacobian(self._f, x, self._dx)
        # linalg.solve(A,B) returns the matrix solution to AX = B, so
        # it gives (A^{-1}) B. np.matrix() promotes scalars to 1x1
        # matrices.
        h = np.linalg.solve(np.matrix(Df_x), np.matrix(fx))
        # Suppose x was a scalar. At this point, h is a 1x1 matrix. If
        # we want to return a scalar value for our next guess, we need
        # to re-scalarize h before combining it with our previous
        # x. The function np.asscalar() will act on a numpy array or
        # matrix that has only a single data element inside and return
        # that element as a scalar.
        if np.isscalar(x):
            h = np.asscalar(h)

        return x - h #this needs to be a subtraction, not addition to satisfy the Newton method
