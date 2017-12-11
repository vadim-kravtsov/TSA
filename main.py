#! /usr/bin/env python
import sys
import os
from os import path
from math import cos, pi
sys.path.append(path.join(os.getcwd(), "lib"))
from lib.tsgen import tsgen
import numpy
from numpy.fft import fft
from scipy import signal
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def plot(x, y, title=''):
	plt.title(title)
	plt.ylabel('y')
	plt.xlabel('Time, s')
	plt.grid(True)
	plt.plot(x,y)
	#plt.savefig('result.png')
	plt.show()

def centering(X):
	m = sum(X)/len(X)
	return [x-m for x in X]

def detrending(x, y):
	model = LinearRegression()
	x = numpy.reshape(x, (len(x), 1))
	model.fit(x, y)
	trend = model.predict(x)
	
	detrended = [y[i]-trend[i] for i in range(0, len(y))]
	return trend, detrended

def periodogram(Y):
	m = len(Y)**2
	D = [1/m*((x.real)**2+(x.imag)**2) for x in Y]
	return D

def autocorr(x):
    result = numpy.correlate(x, x, mode='full')
    return result[int(len(result)/2):]

def Tukey_weight(N1, m):
	a = 0.25
	W = (1.0-2.0*a) + 2*a*cos(pi*m/N1)
	return W

def weighed_correlogramm(N, c):
	N1 = 0.5*N
	wc = [x*Tukey_weight(N1,i) for (i, x) in enumerate(c[:int(N1)])]
	return wc

def main():
	dt = 1
	x, y = tsgen()
	Y = centering(y)
	plt.plot(x, Y)
	trend, Y = detrending(x,Y)
	plt.plot(trend)
	plt.show()
	plt.plot(x, Y)
	plt.show()
	f, ppx = signal.periodogram(Y)
	plt.plot(f, ppx)
	plt.show()
	c = autocorr(Y)
	N = len(c)
	wc = weighed_correlogramm(N,c)
	wc = autocorr(wc)
	#plt.plot(x,c)
	dnu = 1/(dt*len(wc))
	nu = [k*dnu for k in range(len(wc))]
	plt.plot(nu,wc)
	plt.show()
	#for i in range(230):
	#	print(y[i], Y[i])
	#print(type(Y))
	#k=0
	#for i in Y:
	#	k+=1
	#print(k)

if __name__ == '__main__':
	main()