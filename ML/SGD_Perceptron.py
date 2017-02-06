import numpy as np
import matplotlib.pyplot as plt

# import tp1ML

#%% Calcul de la fonction hinge et de son gradient

def hinge(w,data,y,alpha=0,stochastic=False): 	
	if (data.ndim==1):
		d=data.size
		data=data.reshape((1,d))
	(n,d)=data.shape
	w=np.reshape(w,(-1,1))
	y=np.reshape(y,(-1,1))
	if (d != w.shape[0]) or (n != y.shape[0]):
		print(d,"différent de",w.shape[0],"\n")
		print("ou",n,"différent de",y.shape[0],"\n")
	else: 
         classificationFunction=np.dot(data,w)
         hingeLoss=np.maximum(0,alpha-y*classificationFunction)
         if stochastic:
             indice=np.random.randint(n)
             return hingeLoss[indice]
         else:
             averageLoss=np.mean(hingeLoss,axis=0)
             return averageLoss

def hingeGrad(w,data,y,alpha=0,stochastic=False): 
    if (data.ndim==1):
        d=data.size
        data=data.reshape((1,d))
    if (data.ndim!=2):
        print("Wrong dimensions for data")
    else:
        (n,d)=data.shape
        w=np.reshape(w,(-1,1))
        y=np.reshape(y,(-1,1))
        if (d != w.shape[0]) or (n != y.shape[0]):
            print(d,"différent de",w.shape[0],"\n")
            print("ou",n,"différent de",y.shape[0],"\n")
        else:
            classificationFunction=np.sign(y*np.dot(data,w))<0
            hingeGrad=-data*(y*classificationFunction)
            if stochastic:
                indice=np.random.randint(n)
                randomGrad=hingeGrad[indice,:]
                return randomGrad
            else:
                averageGrad=np.mean(hingeGrad,axis=0)
                return averageGrad

#%% Test de hinge et de hingeGrad	

w = np.random.random((4,))*2-1
data1 = np.random.random((100,3))*2-1
data=np.hstack((np.ones((100,1)),data1))
y = np.random.randint(0,2,size = (100,))*2-1
 
#### doit retourner un scalaire
print (hinge(w,data,y), hinge(w,data[0],y[0]), hinge(w,data[0,:],y[0]))
 ### doit retourner un vecteur de taille (w.shape[1],) 
print (hingeGrad(w,data,y))
print(hingeGrad(w,data[0],y[0]),hingeGrad(w,data[0,:],y[0]))

#%% Perceptron

class Perceptron(): 
    def __init__(self,max_iter = 1000):
        self.max_iter = max_iter
    def fit(self,data,y,eps=1e-3,stochastic=False,w=None):
        self.i=0
        if w==None:
            self.w = np.random.random(data.shape[1])
        else:
            self.w=w
        self.eps=eps
        self.dim=(self.w).shape[0]
        self.hist_w = np.zeros((self.dim,self.max_iter)) 
        self.hist_f = np.zeros(self.max_iter)
        while self.i < self.max_iter:  
            self.w = self.w - self.eps*hingeGrad(self.w,data,y,stochastic=stochastic)
            self.hist_w[:,self.i]=np.squeeze(self.w)
            self.hist_f[self.i]=hinge(self.w,data,y,stochastic=stochastic)
#			if self.i % 100==0:
#				print (self.i," itérations")
#				print("La loss vaut ",self.hist_f[self.i]) 
            self.i+=1
    def predict(self,data):
        return np.sign(np.dot(data,self.w))   
    def score(self,data,y):
        return np.mean(y*self.predict(data)>0)

iterations=5000
step=1e-3

classifier=Perceptron(iterations)
classifier.fit(data,y,stochastic=False,w=None)

classifier_sto=Perceptron(iterations)
classifier_sto.fit(data,y,stochastic=True,w=None)
 
x=np.arange(iterations)
plt.figure()
plt.plot()
plt.plot(x,classifier.hist_f,"r")
#plt.plot(x,classifier_sto.hist_f,"b")
plt.title("Fonction loss avec pas={} et {} iterations".format(step,iterations))
plt.show()

#%% Génération de données 2D, plot des données et de la frontière

def gen_arti(centerxPos=1,centeryPos=1,centerxNeg=-1,centeryNeg=-1,sigma=0.1,nbex=1000,data_type=0,eps=0.1):
	""" Générateur de données,
		:param centerx: centre des gaussiennes
		:param centery: 
		:param sigma: des gaussiennes
		:param nbex: nombre d’exemples
		:param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
		:param eps: bruit dans les donnees
		:return: data matrice 2d des donnnes,y etiquette des donnnees 
		""" 
	if data_type==0:
		#melange de 2 gaussiennes 
		xpos=np.random.multivariate_normal([centerxPos,centeryPos],sigma*np.eye(2),nbex//2) 
		xneg=np.random.multivariate_normal([centerxNeg,centeryNeg],sigma*np.eye(2),nbex//2) 
		data=np.vstack((xpos,xneg)) 
		y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2))) 
	if data_type==1:
		#melange de 4 gaussiennes 
		xpos=np.vstack((np.random.multivariate_normal([centerxPos,centeryPos],sigma*np.eye(2),nbex//4),np.random.multivariate_normal([centerxNeg,centeryNeg],sigma*np.eye(2),nbex//4))) 
		xneg=np.vstack((np.random.multivariate_normal([centerxPos,centeryNeg],sigma*np.eye(2),nbex//4),np.random.multivariate_normal([centerxNeg,centerxPos],sigma*np.eye(2),nbex//4))) 
		data=np.vstack((xpos,xneg)) 
		y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))
	if data_type==2:
		#echiquier
		data=np.reshape(np.random.uniform(-4,4,2*nbex),(nbex,2))
		y=np.ceil(data[:,0])+np.ceil(data[:,1])
		y=2*(y % 2)-1 
	# un peu de bruit 
	data[:,0]+=np.random.normal(0,eps,nbex) 
	data[:,1]+=np.random.normal(0,eps,nbex) 
	# on mélange les données 
	idx = np.random.permutation((range(y.size))) 
	data=data[idx,:]
	y=y[idx]
	return data,y

def plot_data(data,labels=None): 
    cols,marks = ["red", "green", "blue", "orange", "black", "cyan"],[".","+","*","o","x","^"] 
    if labels is None:
        plt.scatter(data[:,0],data[:,1],marker="x") 
        return 
    for i,l in enumerate(sorted(list(set(labels.flatten())))):
        plt.scatter(data[labels==l,0],data[labels==l,1],c=cols[i],marker=marks[i])

def make_grid(data=None,xmin=-5,xmax=5,ymin=-5,ymax=5,step=20):
    if data!=None:
        xmax, xmin, ymax, ymin = np.max(data[:,0]), np.min(data[:,0]), np.max(data[:,1]), np.min(data[:,1])
    x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step)) 
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def plot_frontiere(data,f,step=50):
    grid,x,y=make_grid(data=data,step=step) 
    plt.contourf(x,y,f(grid).reshape(x.shape),colors=("gray","blue"),levels=[-1,0,1])

#%% Essai sur des données de type 0
       
iterations=3000
step=1e-1
eps=1e-1

datax,datay = gen_arti(eps=eps,data_type=1)
p = Perceptron(max_iter=iterations)
p.fit(datax,datay,eps=step)
print ("Le score vaut {}".format(p.score(datax,datay))) 
plot_frontiere(datax,p.predict,50)
plt.title("Frontière de décision pour eps={} et pour step={}".format(eps,step))
plot_data(datax,datay)

#%% Essai sur des données de variance diverses pour des pas variés

# sigma_2 vecteur de variances allant de 0.1 à 3
sigma_2=0.1*np.arange(1,30)
sigma_plus=np.squeeze(np.dstack((sigma_2,np.arange(sigma_2.size))))

# steps vecteur de pas d'apprentissages
steps=np.squeeze(np.dstack((np.array([1e-3,1e-2,1e-1]),np.arange(3))))

hist_score=np.zeros((sigma_2.size,3))

# Pour contrebalancer l'aléas d'initialisations différentes (c'est nécessaire pour calculer le score),
# on réalise plusieurs mesures à pas et variance fixées.
nombreMesures=1

w_init=np.random.random(3)
for couple in sigma_plus:
    noiseVariance,i=couple[0],couple[1]
    datax,datay = gen_arti(data_type=0,nbex=1000,eps=noiseVariance,centerxPos=2,centeryPos=2,centerxNeg=0,centeryNeg=0)
    data=np.hstack((np.ones((1000,1)),datax))
    for couple2 in steps :
        learningStep,j=couple2[0],couple2[1]
        mesures=np.zeros(nombreMesures)
        for k in range(nombreMesures): 
            p.fit(data,datay,eps=learningStep,stochastic=False,w=w_init)
            mesures[k]=p.score(data,datay)
        hist_score[i,j]=np.mean(mesures)

fig=plt.figure()
plt.plot(sigma_2,hist_score[:,0],"g",label="step=0.001")
plt.plot(sigma_2,hist_score[:,1],"b",label="step=0.01")
plt.plot(sigma_2,hist_score[:,2],"r",label="step=0.1")      
plt.title("Score d'optimisation pour différentes valeurs du pas pour {} itérations".format(iterations))
plt.xlabel("Variance du bruit")
plt.ylabel("Score d'optimisation")
plt.legend()
plt.show()

#%% Génération des données USPS

def load_usps(filename):
    with open(filename ,"r") as f:  
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)


dataxUSPS,datayUSPS = load_usps("usps.txt")
plt.imshow(dataxUSPS[3].reshape((16,16)),interpolation="nearest")
print(datayUSPS[3])

#%% Séparation de diverses classes

indicesPos=np.where((datayUSPS==1) | (datayUSPS==3) | (datayUSPS==5) | (datayUSPS==7) | (datayUSPS==9))
indicesNeg=np.where((datayUSPS==2) | (datayUSPS==4) | (datayUSPS==6) | (datayUSPS==8))

datayPos=np.ones(indicesPos[0].size)
datayNeg=-np.ones(indicesNeg[0].size)

datax1=np.vstack((dataxUSPS[indicesPos],dataxUSPS[indicesNeg]))
datay1=np.hstack((datayPos,datayNeg))

p2 = Perceptron(max_iter=iterations)
p2.fit(datax1,datay1,eps=step)
print("Le score vaut {}".format(p2.score(datax1,datay1)))

weightVector=p2.w
plt.imshow(weightVector.reshape(16,16),interpolation="nearest")

#%% Projection polynomiale

def poly_proj_2(data):
    n=data.shape[0]
    assert(data.shape[1]==2)
    firstComponent=np.expand_dims(data[:,0],axis=1)
    secondComponent=np.expand_dims(data[:,1],axis=1)
    proj_2=np.hstack((np.ones((n,1)),data,firstComponent**2,secondComponent**2,firstComponent*secondComponent))
    return proj_2

iterations=3000
step=1e-1
eps=1e-1

datax,datay=gen_arti(eps=eps,data_type=1)
p=Perceptron(max_iter=iterations)
newData=poly_proj_2(datax)
p.fit(newData,datay)
print ("Le score vaut {}".format(p.score(newData,datay)))
#plot_frontiere(newData,p.predict,50)
#plt.title("Frontière de décision pour eps={} et pour step={}".format(eps,step))
#plot_data(datax,datay)


#%% Noyau gaussien

def generate_points(B,data_type=2,centerxPos=0,centeryPos=0,var=1):
    dataCenter=np.hstack((centerxPos*np.ones((B,1)),centeryPos*np.ones((B,1))))
#    if (data_type==2):
#    else:
    return dataCenter+var*np.random.rand(B,2)

def gaussian_proj(data,points):
    B=points.shape[0]
    n=data.shape[0]
    reshapePoints=np.tile(np.reshape(points,-1),(1,2*n))
    reshapeData=np.tile(data,2*B)
    reshapePoints1,reshapePoints2=reshapePoints[:,0:2:2*B-2],reshapePoints[:,1:2:2*B-1]
    reshapeData1,reshapeData2=reshapeData[:,0:2:2*B-2],reshapeData[:,1:2:2*B-1]
    phiB=np.exp(-((reshapePoints1-reshapeData1)**2+(reshapePoints2-reshapeData2)**2))
    return phiB

B=50
iterations=3000
step=1e-1
eps=1e-1
    
p2 = Perceptron(max_iter=iterations)
pointsReference=generate_points(B)
datax,datay = gen_arti(eps=eps,data_type=0)
p2.fit(datax,datay,eps=step)
print ("Le score vaut {}".format(p2.score(datax,datay)))
#plot_frontiere(datax,p2.predict,50)
#plt.title("Frontière de décision pour eps={} et pour step={}".format(eps,step))
#plot_data(datax,datay)
