import numpy as np
import matplotlib.pyplot as plt


r = 0.1
sigma = 0.3
T = 1
S0 = 100

##### Calcul de l'espérance

def esperance(r = 0.1, sigma = 0.3, T = 1, S0 = 100):
	return ((np.exp(r*T)-1)*S0)/(r*T)



##### Calcul de la variance

def variance(r = 0.1, sigma = 0.3, T = 1, S0 = 100):
	temp1 = (2*S0**2/((T**2)*(r+sigma**2)))
	temp2 = (np.exp((2*r+sigma**2)*T)-1)/(2*r+sigma**2)
	temp3 = (np.exp(r*T)-1)/r
	temp = temp1 * (temp2 - temp3)
	return temp - (((np.exp(r*T)-1)*S0)/(r*T))**2

def ecarttype(r = 0.1, sigma = 0.3, T = 1, S0 = 100):
	return np.sqrt(variance(r, sigma, T, S0))


esp = esperance()
var = variance()
EC = ecarttype()

mulog = np.log(esp**2/np.sqrt(var + esp**2))
sigmalog = np.log(1 + (var/(esp**2)))


##### Monte Carlo pour lognormale

##e K = 0.9 S0

def Monte_Carlo(iterations = 10, K = 90, T = 1, r = 0.1):
    lognorm = np.random.lognormal(mulog, sigmalog, iterations)
    Price = np.exp(-r*T) * np.maximum(lognorm - K, 0)
    std = np.std(Price)
    mean = np.mean(Price)
    ##print(Price, std, mean)
    return mean+(1.96*std/np.sqrt(iterations)), mean, mean-(1.96*std/np.sqrt(iterations))


L = list(range(200))
L = [e*50  for e in L]


## Convergence en racine n.
Monte_Carlo_u = [Monte_Carlo(iterations=e) for e in L]
t = plt.plot(L, Monte_Carlo_u)
plt.legend(iter(t), ( 'Upper confident interval', 'Price', 'Lower confident interval'))
plt.xlabel('iterations')
plt.ylabel('Expected Price')
plt.show()





##### Monte Carlo en discrétisant












