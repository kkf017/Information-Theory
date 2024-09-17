import numpy

from dataclasses import dataclass
from typing import List


# https://engineering.purdue.edu/kak/Tutorials/ExpectationMaximization.pdf
# https://www.youtube.com/playlist?list=PLBv09BD7ez_4e9LtmK626Evn1ion6ynrt
# https://medium.com/@clarice.wang/understanding-the-em-algorithm-by-examples-with-code-and-visualization-dc93657adc84
# https://medium.com/b2w-engineering-en/the-reasoning-behind-the-expectation-maximization-em-algorithm-4a773428b3fc

# https://medium.com/b2w-engineering-en/the-reasoning-behind-the-expectation-maximization-em-algorithm-4a773428b3fc

# https://medium.com/@zhe.feng0018/coding-gaussian-mixture-model-and-em-algorithm-from-scratch-f3ef384a16ad

# https://ics.uci.edu/~smyth/courses/cs274/notes/notes2022/mixture_models_EM.pdf

# https://www.geeksforgeeks.org/covariance-matrix/
# https://datascienceplus.com/understanding-the-covariance-matrix/


N = 2 # number of gaussian mixture models



@dataclass
class GaussianModel:
	mu = None # mean 
	sigma = None # standard deviation / covariance
	weights = None # mixture weights, such as P(z = c)



def dataset():
	""" Function that creates a dataset with n gaussian ditribution N(mu, sigma)."""
	pass


def gaussian(mu, sigma, n):
	""" Function that generates random gaussian data. """
	return numpy.random.normal(mu, sigma, n)
	

def mean(X):
	""" Function that computes mean of a dataset. """
	return numpy.mean(X, axis=0)


def covariance(X):
	""" Function that computes covariance of a dataset. """
	def cov(x, y):
		return numpy.sum(x*y)/(x.shape[0]-1)
	
	n, m = X.shape[0], X.shape[1]
	mu = numpy.mean(X, axis=1)
	x = [ [cov(X[i]-mu[i], X[j]-mu[j]) for j in range(n)] for i in range(n)]
	return numpy.array(x)



	


def EM(X, n):
	"""
	1. Randomly initialize μ's, Σ's, and π's;
	2. E-Step:
    		Compute the responsibilities rij for every data point xᵢ using μ, Σ, and π, and the equation (10)
    	
    	3. M-Step:
    		Update the parameters μ, Σ, and π (plugging the responsibilities evaluated in the E-Step) using the set of equations (12-14)
    		
    		4. Repeat steps 2 and 3, until there is no significant change in the proxy function of the log-likelihood given by equation (15)
	"""
	
	def Normal(x, mu, sigma):
		""" Function that computes Mulivariate Gaussian model N(x;mu, sigma)."""
		d = x.shape[0]
		const = (1 / (2*numpy.pi)**(d/2)) * numpy.linalg.det(sigma)**(-1/2)
		prod = numpy.dot(numpy.dot((x-mu), numpy.linalg.inv(sigma)), (x-mu).T)
		return const * numpy.exp(-1/2 * prod)
	
	def initialize(k, n):
		""" Function that initializes μ's, Σ's, and π's for clusters. """
		clusters = []
		for i in range(n):
			cluster = {}
			cluster["mu"] = numpy.random.normal(0,1, (k))
			cluster["sigma"] = numpy.eye(k)
			cluster["weights"] = 1/k
			clusters.append(cluster)
		return clusters
	
	def E(X, clusters):
		""" Function that implements E step. """
		ric = numpy.zeros((X.shape[0], len(clusters)))
		for i in range(X.shape[0]):
			for j, cluster in enumerate(clusters):
				ric[i,j] = cluster["weights"] * Normal(X[i], cluster["mu"], cluster["sigma"])
				
		for i in range(X.shape[0]):
			ric[i, :] = ric[i, :] / numpy.sum(ric[i, :])
		print(f"\n\nric: \n{ric}")
		return ric
					
	
	def M(X, clusters, ric):
		""" Function that implements M step. """
		mc = numpy.sum(ric, axis=0)
		print(f"\n\nmc: {mc}, {X.shape[0]}")
		for j, cluster in enumerate(clusters):
			cluster["mc"] = mc[j]
			cluster["weights"] = mc[j] / X.shape[0] # pic = mc /m  = m, N points xi
			print(f"\n\ncluster {j}, {mc[j]}, {mc[j] / X.shape[0]}")
			
			cluster["mu"] = (1 / mc[j]) *  numpy.sum(numpy.multiply(X.T, ric[:,j]).T, axis=0)
			input()
			
			#cluster["sigma"] = 
			clusters[j] = cluster
		return cluster
		
	clusters = initialize(X.shape[1], n)
	print(f"\n\n{clusters}")
	ric = E(X, clusters)
	clusters = M(X, clusters, ric)
	return None



if __name__ == "__main__":

	mean1, mean2 = numpy.array([0, 0]), numpy.array([10, 20])
	sigma1, sigma2 = numpy.array([[1, 0], [0, 1]]), numpy.array([[5, -5], [-5, 10]])

	x1 = numpy.random.multivariate_normal(mean1, sigma1, 5) 
	x2 = numpy.random.multivariate_normal(mean2, sigma2, 2)
	x3 = numpy.random.multivariate_normal(mean1, sigma1, 2)
	x4 = numpy.random.multivariate_normal(mean2, sigma2, 5)
	
	x = numpy.concatenate((x1, x2, x3, x4), axis=0)
	
	print(x.shape)
	print(x)
	
	EM(x, 2)
	
	"""	
	print("\n",covariance(X))
	print("\n",numpy.cov(X))
	"""
