"""--------------------------------------------
	Entropy and Detection Theory
-----------------------------------------------"""
import numpy as np
from math import log
import scipy.integrate as integrate

import matplotlib.pyplot as plt

# PARAMETERS
SIGMA = 1
P_AWGN  = 0.3
k = 10000
gamma = 1


## General idea
# 1.2. Find the minimum required sample for the detector
#Q(gamma / (sqrt(sigma**2 / N))) = integrale : gamma / (sqrt(sigma**2 / N)) -> inf : 1/sqrt(2*pi) * exp(-1/2 * t**2)
N= 1
a = gamma / (np.sqrt(SIGMA*2 / N))
pf = integrate.quad(lambda t: 1/np.sqrt(2*np.pi)*np.exp(-1/2*t**2), a, np.inf)
while pf[0] > 0.001:
	N+=1
	a = gamma / (np.sqrt(SIGMA*2 / N))
	pf = integrate.quad(lambda t: 1/np.sqrt(2*np.pi)*np.exp(-1/2*t**2), a, np.inf)
	#print(N, pf)
print('Optimum samples N', N)

# Generate input of length k
signal = np.random.randint(0,2, k)
print('Original signal',signal.shape, signal)

# generate voltage signal - duration of each state is N samples, each interval og length N - 
for i in range(signal.shape[0]):
	if signal[i] == 0:
		sample = np.zeros(N)
	if signal[i] == 1:
		sample = np.ones(N)
		
	if i == 0 :
		input = sample
	else:
		input = np.concatenate((input, sample))	
print('Input signal',input.shape)
	
# add white gaussian noise
output = input.copy()
AWGN = np.random.normal(0,SIGMA,input.shape)
AWGN = np.where(abs(AWGN) < P_AWGN)
output[AWGN]  = 1 - output[AWGN] 


# 1.3. Signal detection
H0 = 'H0: x[n]=w[n] (noise)'
H1 = 'H1: x[n]=s[n]+w[n] (signal + noise)'
A = np.ones(N)

# compute T(x) = xT.s > sigma ** 2 * log(gamma) + 1/2 * sTs
y = np.zeros(int(output.shape[0]//N))
for i in range(int(output.shape[0]//N)):
	x = output[i*N : (i+1)*N]
	Tx = np.dot(x,A.T)
	const = SIGMA**2 * log(gamma) + 1/2 * np.dot(A,A.T) 
	#print(Tx,'> or <', const)
	if Tx > const:
		#print(H1, '=> 1')
		y[i] = 1
	if Tx < const:
		#print(H0, '=> 0')
		y[i] = 0
print('bits errors',sum(np.where(signal != y, 1, 0)) / signal.shape[0], '%')


# plot x[n], y[n] and x^[n]
fig, axs = plt.subplots(3)
fig.suptitle('Respectively original x, x^ and y')
axs[0].plot(signal, '.b')
axs[1].plot(y, '.r')
axs[2].plot(output, '.k')
plt.show()


# compute PFA, PM and Pe
PFA = integrate.quad(lambda t: 1/np.sqrt(2*np.pi)*np.exp(-1/2*t**2), gamma, np.inf)
PM = integrate.quad(lambda t: 1/np.sqrt(2*np.pi)*np.exp(-1/2*(t-1)**2), gamma, np.inf)
print('PFA',PFA[0], 'PM', PM[0], 'PE',PM[0]+PFA[0])


