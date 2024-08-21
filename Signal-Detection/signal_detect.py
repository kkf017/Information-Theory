import random
import math

from typing import List

# Rename github-Repositories -> Repository-1, Repository-2, ....etc

# H0 : x[n] = w[n] n=0,...,N-1 (noise)
# H1 : x[n] = s[n] + w[n] n=0,...,N-1 (signal + noise)


AWGN = 0.3
N = 25
GAMMA = 1

def generate_signal(n:int)->List[int]:
	""" Function that creates a random binary signal. """
	return [random.randint(0,1) for _ in range(n)]

def digitalize(x:List[int], n:int)->List[int]:
	"""Function that digitalize a signal. """
	return [i for i in x for _ in range(n)]	

def gaussian_noise(x:List[int], mu:float, sigma:float):
	""" Function that adds gaussian noise to binary signal. """
	return [abs(int(abs(random.gauss(mu, sigma)) < AWGN) - x[i]) for i in range(len(x))]


def Neyman_Pearson(x:List[int], gamma:float, sigma:float)->int:
	""" Function for binary hypothesis testing (with Neyman Pearson coefficient). """
	sign = [1 for _ in range(len(x))]
	f = lambda x,y : x*y
	const = math.log(gamma) * sigma**2 + 1/2 * sum(list(map(f, sign, sign)))
	T = sum(list(map(f, x, sign)))
	if T > const:
		return 1
	else:
		return 0	
				

def signal_detection(x:List[int], gamma:float, sigma:float, n:int)->List[int]:
	""" Function to detect signal. """
	return [Neyman_Pearson(x[i:i+n], gamma, sigma) for i in range(0,len(x),n)]



if __name__ == "__main__":
	
	x = generate_signal(1000)
	y = digitalize(x, N)
	gauss = gaussian_noise(y, 0, 1)	
	sign = signal_detection(gauss, GAMMA, 1, N)
		
	err = sum([int(x[i] != sign[i]) for i in range(len(x))])
	print(f"Error: {err} / {len(x)} = {err/len(x)} %")
	
