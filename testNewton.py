#!/usr/bin/env python3

import unittest
import numpy as np

import newton

class TestNewton(unittest.TestCase):
    def testLinear(self):
        # Just so you see it at least once, this is the lambda keyword
        # in Python, which allows you to create anonymous functions
        # "on the fly". As I commented in testFunctions.py, you can
        # define regular functions inside other
        # functions/methods. lambda expressions are just syntactic
        # sugar for that.  In other words, the line below is
        # *completely equivalent* under the hood to:
        #
        # def f(x):
        #     return 3.0*x + 6.0
        #
        # No difference.
        f = lambda x : 3.0*x + 6.0

        # Setting maxiter to 2 b/c we're guessing the actual root
        solver = newton.Newton(f, tol=1.e-15, maxiter=4)
        x = solver.solve(-2.0)
        # Equality should be exact if we supply *the* root, ergo
        # assertEqual rather than assertAlmostEqual
        self.assertEqual(x, -2.0)
        return
    
    '''Newton's law has a few functions that won't work with just this code:
        1. If no root exists for the function (y=1,x2+1,y=e^x)
        2. poor initial guessing could end up in an infinite loop around the 
            wrong value
        3. If there is an inflection point and the initial guess is directly on
            the vertex (for quadratics and x^3)
        4. If the derivative does not exist at the root
        5. The derivative is not a continuous function
    '''
    def test_there_are_no_roots(self):
        f = lambda x : 0*x + 13.5
        #f = lambda x : x*x + 1.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(-1)
        #it should be exactly equal if the root is provided
        #i = np.sqrt(-1)
        self.assertEqual(x,-13.5)
        return
    
    def test_poor_initial_guess(self):
        import math
        f = lambda x : x*math.exp(-x*x)
        #if maxiter is 20 here, the value given back is not the root, it is
        #just the value that the solver ended up on
        #when maxiter is 2000, gives 6.06
        solver = newton.Newton(f, tol=1.e-15, maxiter=2000)
        x = solver.solve(1)
        self.assertAlmostEqual(x, 0)
        return
    
    def test_higher_order(self):
        f = lambda x : x*x - 1
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(-0.01)
        self.assertAlmostEqual(x, 1.0)
        #so this finds one of the roots, but how do we find the other? 
        #need to develop a way to get second root that is opposite
        return
    
    def test_no_deriv_root(self):
        f = lambda x : x*x
        solver = newton.Newton(f, tol=1.e-15, maxiter = 10)
        x = solver.solve(3)
        self.assertAlmostEqual(x, 0)
        return
    
    def test_divergent_function(self):
        f = lambda x : 1/(2+x)
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(0)
        self.assertAlmostEqual(x,0)
        return
        
if __name__ == "__main__":
    unittest.main()

    
