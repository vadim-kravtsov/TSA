from math import sin, cos, sqrt, pi
from random import normalvariate
import matplotlib.pyplot as plt

def tsgen(N = 230, dt = 1, A = 1, ab = (0.1, 0.05), nufi = (0.1, 0), gamma = 0.5):
	'''
	A function that generates a time series.
	'''
	sigma = sqrt(A**2/(2*gamma))
	a, b = ab
	nu, fi = nufi
	x = [[],[]]
	for k in xrange(N):
		t = dt*k
		zeta = normalvariate(0.0, 1.0)
		print zeta
		x[0].append(t)
		x[1].append(a + b*t + A*cos(2*pi*nu*t - fi) + sigma*zeta)
	return x

if __name__ == '__main__':
	x, y = tsgen()
	plt.title('The graph of generated time serial:')
	plt.ylabel('y')
	plt.xlabel('Time, s')
	plt.grid(True)
	plt.plot(x,y)
	plt.savefig('result.png')
	plt.show()