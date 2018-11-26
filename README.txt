
Karina Alventosa

Program to find roots of various functions using Newton's method

Newton(object) is the class designed to contain the solve() method for approximating roots from 1D or 2D functions. 

It contains 4 functions:

__init__: the constructor method for the class, it allows for the class to initialize its attributes 
	Parameters:
	f: the function that will be assessed to find the roots
	tol: tolerance to determine how close the derivative must be to 0 to accept the
		value as an acceptable root
	maxiter: the number of specified iterations that the solver will perform in the attempt to attain the root
		 Using Newton's method, most will converge rapidly and require few iterations
	dx: step size required for computing the numerical Jacobian
	radius_max: binds the root to a specified range around the initial guess to avoid any issues with the solver
		    if the method causes a loop or cannot converge
	Df: analytical Jacobian to provide a possibly more efficient alternative to numerical approximations

solve: the function that numerically calculates the derivative of the function and determines whether it is within the specified tolerance and binding radius. Using the following line of code, it calculates the formula for Newton's method and checks whether it is within tolerance. If it is, the loop stops and x is returned. If not, the loop is stepped and the iterations proceed with the new slope of the previously calculated tangent line.
	example: if np.linalg.norm(fx) < self._tol and self._r is None

	Solve also contains a checker for the case in which there are no roots, or when maxiter as been reached. This 
	can be applicable to poor initial guesses as well. 

step: this function generates the stepping of the Newton's method for the solve function. It also calculates the derivative numerically for the specified function and returns the function multiplied by the inverse of its derivative as required by the Newton's method equation. 

Checker: feature that allows the user to add any additional checks that may be required when singular issues arise. It is currently implementing the check for whether the root calculations have remained within the maximum radius specified. This function can be expanded to include any error or exception messages for other unique cases that will be discovered when utilizing this function in a more in-depth manner. 

The file functions.py contains a function that calculates the numerical Jacobian (approximateJacobian(f, x, dx=1e-6)) for the newton class, as well as contains a Polynomial class that can automatically construct polynomials for the solver to find the roots of. 

testNewton contains a series of tests to determine the efficacy of the newton solver. All have been added to allow the solver to work in a variety of cases in which Newton's method fails or is not efficient. 

testFunctions contains tests to ensure that the numerical approximation of the Jacobian and the polynomial generator are performing correctly. 