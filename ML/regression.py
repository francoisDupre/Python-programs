#%%
from tools import *
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import sklearn as sl
import random
import os


mpl.rcParams['legend.fontsize'] = 10
#%% Classe OptimFunc

class OptimFunc:
	def __init__(self,f=None,grad_f=None,dim=2):
		self.f=f
		self.grad_f=grad_f
		self.dim=dim
	def init(self,low=-1,high=1):
		return np.random.rand(self.dim)*(high-low)+low

def lin_f(x): return x
def lin_grad(x): return 1
lin_optim=OptimFunc(lin_f,lin_grad,1)
lin_optim.f(3)
lin_optim.grad_f(1)

def xcosx(x): return x*np.cos(x)
def xcosx_grad(x): return np.cos(x)-x*np.sin(x)
xcosx=OptimFunc(xcosx,xcosx_grad,1)

def rosen(x1,x2): return 100*(x2-x1**2)**2 + (1-x1)**2
def rosen_grad(x1,x2):
	f1=400*x1*(x1**2-x2) + 2*(x1-1)
	f2=200*(x2-x1**2)
	return np.array([f1,f2])
	
rosen_f=OptimFunc(rosen,rosen_grad,2)


#%% Classe GradientDescent


class GradientDescent: 
    def __init__(self,optim_f,max_iter=5000,i=0):
        self.optim_f=optim_f
        self.max_iter=max_iter
        self.i=i
    def reset(self): 
        self.i=0
        self.w = self.optim_f.init() 
    def optimize(self,w=None,refresh=True,eps=1e-3):
        self.eps=eps
        if refresh: 
            self.reset()
        else:
            self.w=w_init
        self.log_w = np.array(self.w) 
        self.log_f = np.array((self.optim_f).f(*self.w)) 
        self.log_grad = np.array((self.optim_f).grad_f(*self.w)) 
        while not self.stop(): 
            self.w = self.w - self.eps*self.optim_f.grad_f(*self.w) 
            self.log_w=np.vstack((self.log_w,self.w)) 
            self.log_f=np.vstack((self.log_f,(self.optim_f).f(*self.w))) 
            self.log_grad=np.vstack((self.log_grad,(self.optim_f).grad_f(*self.w))) 
            self.i+=1
        print("On a effectué", self.i, "iterations. \n")
#        print("On est placé en (",self.log_w[self.i,0],",",self.log_w[self.i,1],")")
#        print("La valeur finale de la fonction est", self.log_f[self.i][0])
        return (self.log_w,self.log_f,self.log_grad)
    def stop(self): 
        return (self.i>2) and (self.i>self.max_iter)
    def get_eps(self): 
        return self.eps

#%% Affichage des fonctions 1D et Rosen


xrange=np.arange(-5,5,0.1) 
plt.plot(xrange,xcosx.f(xrange)) 
plt.show()

fig = plt.figure()
ax=fig.gca(projection='3d')
X=np.arange(-5,5,0.1)
Y=np.arange(-5,5,0.1)
X,Y=np.meshgrid(X,Y)
Z=rosen_f.f(X,Y)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf)
plt.show()


#%% Optimisation des fonctions avec GradientDescent

nbIterations=5000

fonction=rosen_f

w_init=2*np.random.rand(fonction.dim)-1

fonction_optim_3 = GradientDescent(fonction,max_iter=nbIterations)
(log_w_3,log_f_3,log_grad_3)=fonction_optim_3.optimize(refresh=False,eps=1e-3,w=w_init)
fonction_optim_4= GradientDescent(fonction,max_iter=nbIterations)
(log_w_4,log_f_4,log_grad_4)=fonction_optim_4.optimize(refresh=False,eps=5e-4,w=w_init)
fonction_optim_5 = GradientDescent(fonction,max_iter=nbIterations)
(log_w_5,log_f_5,log_grad_5)=fonction_optim_5.optimize(refresh=False,eps=1e-4,w=w_init)

    


#%% Score d'optimisation de la descente de gradient
fig2=plt.figure()
axes=plt.gca()
iterations=np.arange(nbIterations+2)
plt.plot(iterations,log_f_3,"r",label="eps=0.001")
plt.plot(iterations,log_f_4,"b",label="eps=0.0005")
plt.plot(iterations,log_f_5,"g",label="eps=0.0001")
plt.title("Descente de gradient avec nbIterations={}".format(nbIterations))
plt.xlabel("Iterations")
plt.ylabel("Fonction")
plt.legend()
borne_sup=1

if (nbIterations>=20000):
    borne_sup=0.1
axes.set_xlim([0,nbIterations])
axes.set_ylim([0,borne_sup])
plt.show()


#%% Trajectoire suivie par la descente de gradient

fig3=plt.figure()
axes=plt.gca()
plt.plot(log_w_3[:,0],log_w_3[:,1],"r",label="eps=0.001")
plt.plot(log_w_4[:,0],log_w_4[:,1],"b",label="eps=0.0005")
plt.plot(log_w_5[:,0],log_w_5[:,1],"g",label="eps=0.0010")
plt.title("Trajectoire d'optimisation pour ROSEN avec nbIterations={}".format(nbIterations))
plt.xlabel("x")
plt.ylabel("y")
plt.legend()


#%% Régression Linéaire

nbIterations=5000
iterations=np.arange(nbIterations+2)

def gen_1d(n,eps):
    x=np.random.random(size=(n,1))
    y=2*x+1+eps*np.random.normal(size=(n,1))
    return x,y

class Regression(GradientDescent):
    def __init__(self,max_iter=nbIterations,dim=1):
        self.dim=dim
        self.errorFunc=OptimFunc(self.error,self.grad_error,self.dim)
        GradientDescent.__init__(self,self.errorFunc,max_iter)
        self.data=self.y=self.n=self.w=None
    def fit(self,data,y,eps=5e-2):
        self.y=y
        self.n=y.shape[0]
        self.dim=data.size/self.n+1
        self.data=data.reshape((self.n,self.dim-1))
        self.data=np.hstack((np.ones((self.n,1)),self.data))
        return self.optimize(eps=eps)
    def error(self,w):
        return 0.5*np.mean((np.dot(self.data,w)-self.y)**2)
    def grad_error(self,w):
        intermediate=np.dot(self.data,w)-np.squeeze(self.y)
#        print((self.y).shape)
        return np.mean(intermediate*np.transpose(self.data),axis=1)
    def init(self):
        return np.random.random(self.dim)*(np.max(self.data)-np.min(self.data))+np.min(self.data)
    def predict(self,data):
        (n,dim)=data.shape
        return np.hstack((np.ones((n,1)),data)).dot(self.w)

#%% Optimisation de la droite de fit	
 
n=100
sigma=0.1 

(x,y)=gen_1d(n,sigma)
regression=Regression(dim=2)
(log_w,log_f,log_grad_f)=regression.fit(x,y)

    
testSet=np.expand_dims(np.linspace(0,1,100),axis=1)
predictValues=regression.predict(testSet)

plt.plot(x,y,"r+",label="Train set")
plt.plot(testSet,predictValues,"b",label="Test set")
plt.title("Régression linéaire avec {} points et un bruit de variance {}".format(n,sigma))
plt.legend()
plt.show()

