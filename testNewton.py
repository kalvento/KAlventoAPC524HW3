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
    #this test sets up a 2D funciton and tests the solver's ability to find
    #its roots
    def test_2D(self):
        A = np.matrix("0.0 4.0; 1.0 0.0")
        k = lambda x : A*x
        solver = newton.Newton(k, tol=1.e-15, maxiter=10)
        x = solver.solve(np.matrix("3.0; 15.0"))
        np.testing.assert_almost_equal(x, np.matrix("0.0; 0.0"))
        #so this finds one of the roots, but how do we find the other? 
        #need to develop a way to get second root that is opposite
        return
    
    #This test sets up a more complicated 2D function and sees if the solver
    #can find the roots. They are simplistic for the purposes of the test
    def test_2D_analytical_jacobian(self):
        #lambda setup does not allow for this type of manipulation, so using the
        #old school way of setting up the function like demonstrated above
        def k(x) :
            k = np.matrix("1.0 -4.0; 1.0 1.0")
            k[0,0] *= x[0]*x[0]
            k[0,1] *= x[1]
            k[1,0] *= x[0]*x[0]
            k[1,1] *= x[1]*x[1]
            k1 = np.sum(k[0,:]) #need to sum the rows to get vector valued func.
            k2 = np.sum(k[1,:])
            k = np.matrix([[k1],[k2]])
            return k
        #this sets up the derivative analytically instead of using the numerical
        #solver
        def Dk(x):
            Dk = np.matrix("2.0 -4.0; 2.0 2.0")
            Dk[0,0] *= x[0]
            Dk[1,0] *= x[0]
            Dk[1,1] *= x[1]
            return Dk
        solver = newton.Newton(k, tol=1.e-15, maxiter=1000, Df=Dk)
        x = solver.solve(np.matrix("1.0; 4.0"))
        np.testing.assert_almost_equal(x, np.matrix("0.0; 0.0"))
        return
    #this test will figure out if there are any roots of the function that is
    #provided. If no real roots exist, it will raise an exception
    def test_there_are_no_roots(self):
        f = lambda x : x**2 + 13
        #f = lambda x : x*x + 1.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        #it should be exactly equal if the root is provided
        #i = np.sqrt(-1)
        self.assertRaises(Exception,solver.solve, -1)
        return
    
    #this is testing the result of a poor initial guess and is fixed in newton
    def test_poor_initial_guess(self):
        import math
        f = lambda x : x*math.exp(x)
        solver = newton.Newton(f, tol=1.e-15, maxiter=3000)
        x = solver.solve(-50)
        self.assertAlmostEqual(x, 0)
        return
    
    #test for ensuring the guess is within the reasonable bound
    def test_radius_max(self):
        import math
        f = lambda x : x*math.exp(x)
        solver = newton.Newton(f, tol=1.e-15, maxiter=30, radius_max=0.3)
        self.assertRaises(Exception, solver.solve, -1)
        return   
    
    #this tests the case of a horizontal slope at the first initial guess
    # will require some form of movement of the initial guess to find at least
    # one of the roots
    def test_zero_deriv(self):
        import math
        f = lambda x : -x**2 + 3
        Df = lambda x : -2*x
        solver = newton.Newton(f, tol=1.e-15, maxiter = 30, Df=Df)
        x = solver.solve(0)
        self.assertAlmostEqual(x, -math.sqrt(3))
        return
    
    #similar to the no roots function test
    def test_divergent_function(self):
        f = lambda x : 1/(2+x)
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(0)
        self.assertAlmostEqual(x,0)
        return
    #test for the analytical jacobian feature in 1D
    def test_analytical_jacobian(self):
        f = lambda x : (x-5)**2
        Df = lambda x : 2*(x-5)
        solver = newton.Newton(f, tol=1.e-15, maxiter=30, Df=Df)
        x = solver.solve(0)
        self.assertAlmostEqual(x,5.0)
        return
        
if __name__ == "__main__":
    unittest.main()

    
