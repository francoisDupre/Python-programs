import numpy as np 
# module pour les outils mathématiques 
import matplotlib.pyplot as plt 
# module pour les outils graphiques 
import tools 
# module fourni en TP1 
from sklearn import tree 
# module pour les arbres
#from sklearn import ensemble
# module pour les forets 
from sklearn import cross_validation as cv 
from sklearn.externals.six import StringIO 

#from IPython.display import Image 
import pydotplus
import pickle
import TP3ML
#%%
#Initialisation 
trainData,trainY=tools.gen_arti()
mytree=tree.DecisionTreeClassifier() #creation d’un arbre de decision 
mytree.max_depth=2 #profondeur maximale de 5 
mytree.min_samples_split=1 #nombre minimal d’exemples dans une feuille 
#Apprentissage 
mytree.fit(trainData,trainY)

#prediction 
testData,testY=tools.gen_arti()
pred=mytree.predict(testData)
print("precision : {0}".format((1.*pred!=testY).sum()/len(testY)))

#ou directement pour la precision : 
scoreTest=mytree.score(testData,testY)
print("precision (score) : ".format(scoreTest))

fig1=plt.figure(1)
plt.plot(trainData[:,0],trainData[:,1],"b+")
plt.title("Dataset")
plt.show()

#Importance des variables : 
fig2=plt.figure(2)
a=mytree.feature_importances_
plt.bar([1,2],a)
plt.title("Importance Variable")
plt.xticks([1,2],["x1","x2"])

#Affichage de l’arbre

#dot_data = StringIO() 
#tree.export_graphviz(mytree, out_file=dot_data) 
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf("tree.pdf") 

def affiche_arbre(tree):
    long = 10 
    sep1="|"+"-"*(long-1)
    sepl="|"+" "*(long-1)
    sepr=" "*long 
    def aux(node,sep): 
        if tree.tree_.children_left[node]<0:
            ls ="(%s)" % (", ".join( "%s: %d" %(tree.classes_[i],int(x))
for i,x in enumerate(tree.tree_.value[node].flat))) 
            return sep+sep1+"%s\n" % (ls,)
        return (sep+sep1+"X%d<=%0.2f\n"+"%s"+sep+sep1+"X%d>%0.2f\n"+"%s" )% \
(tree.tree_.feature[node],tree.tree_.threshold[node],aux(tree.tree_.children_left[node],sep+sepl),
tree.tree_.feature[node],tree.tree_.threshold[node],aux(tree.tree_.children_right[node],sep+sepr))
    return aux(0,"")

print(affiche_arbre(mytree))

#with open("mytree.dot","w") as f:
#    tree.export_graphviz(mytree,f)
#    os.unlink(f)

#    Image(graph.create_png())


#%%

def load_usps(filename):
    with open(filename ,"r") as f:  
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)
data1,y1 = load_usps("usps.txt")
N=data1.shape[0]
data=data1
y=y1

split=0.3

#data,y=tools.gen_arti(data_type=2)
#permet de partager un ensemble en deux ensembles d'apprentissage et de test 
data_train,data_test,y_train,y_test=cv.train_test_split(data,y,test_size=0.3)
mytree=tree.DecisionTreeClassifier() #creation d’un arbre de decision 
mytree.max_depth=26 #profondeur maximale de 5 
mytree.min_samples_split=12 #nombre minimal d’exemples dans une feuille 

#alternative : obtenir les indices et itérer dessus  
kf= cv.KFold(y.size,n_folds=10)
res_train=[]
res_test=[]
for cvtrain,cvtest in kf:
    mytree.fit(data[cvtrain],y[cvtrain])
    res_train+=[mytree.score(data[cvtrain],y[cvtrain])]
    res_test+=[mytree.score(data[cvtest],y[cvtest])]
print("moyenne train : {0}".format(np.mean(res_train)))
print("moyenne test : {0}".format(np.mean(res_test)))

plt.imshow(np.reshape(mytree.feature_importances_,(16,16)))

             

#%%

# USPS : 1..4..21, 1..51..101

split=0.3
depthMin=10
depthMax=41
depthStep=4
splitMin=12
splitMax=13
splitStep=2

# depth=11,split=12
# depth=10, split=12 0.88
# depth=20, split=6 0.892
# depth=30, split=12
#split=12, depth=14,26,36 0.894

#permet de partager un ensemble en deux ensembles d’apprentissage et de test 

data_train,data_test,y_train,y_test=cv.train_test_split(data,y,test_size=split) 

#alternative : obtenir les indices et itérer dessus 
kf= cv.KFold(y.size,n_folds=10)
depths=np.arange(depthMin,depthMax,4)
l1=depths.size
splits=np.arange(splitMin,splitMax,2)
l2=splits.size
averageTrainError=np.zeros((l1,l2))
averageTestError=np.zeros((l1,l2))
for cvtrain,cvtest in kf: 
    for j,depth in zip(np.arange(0,l1),depths):
        for i,samplesSplit in zip(np.arange(0,l2),splits):
            mytree=tree.DecisionTreeClassifier() #creation d’un arbre de decision 
            res_train=[] 
            res_test=[] 
            mytree.max_depth=depth #profondeur maximale de 5
            mytree.min_samples_split=samplesSplit # nombre minimal d’exemples dans une feuille 
            mytree.fit(data[cvtrain],y[cvtrain]) 
            res_train+=[mytree.score(data[cvtrain],y[cvtrain])] 
            res_test+=[mytree.score(data[cvtest],y[cvtest])]
            averageTrainError[j,i]=np.mean(res_train)
            averageTestError[j,i]=np.mean(res_test)

# augmenter l'écart entre les différentes splits :

#fig1=plt.figure(1)
#axes=plt.gca()
#depth1,depth2,depth3,depth4,depth5=0,1,2,3,4
#plt.plot(splits,averageTestError[depth1,:],"r",label="depth={0}".format(depths[depth1]))
#plt.plot(splits,averageTestError[depth2,:],"b",label="depth={0}".format(depths[depth2]))
#plt.plot(splits,averageTestError[depth3,:],"g",label="depth={0}".format(depths[depth3]))
#plt.plot(splits,averageTestError[depth4,:],"black",label="depth={0}".format(depths[depth4]))
#plt.plot(splits,averageTestError[depth5,:],"black",label="depth={0}".format(depths[depth5]))
#plt.title("Test error (dataset 2) for differents depths")
#plt.xlabel("Minimum number of samples per split")
#plt.ylabel("Test error")
#plt.show()

fig1=plt.figure(2)
axes=plt.gca()
split1,split2,split3,split4,split5=0,1,2,3,4
plt.plot(depths,averageTestError[:,split1],"r",label="sample_split={0}".format(splits[split1]))
#plt.plot(depths,averageTestError[:,split2],"b",label="sample_split={0}".format(splits[split2]))
#plt.plot(depths,averageTestError[:,split3],"g",label="sample_split={0}".format(splits[split3]))
#plt.plot(depths,averageTestError[:,split4],"black",label="sample_split={0}".format(splits[split4]))
#plt.plot(depths,averageTestError[:,split5],"black",label="sample_split={0}".format(splits[split5]))
plt.title("Test error (dataset 2) for differents samples splits")
plt.xlabel("Depth")
plt.ylabel("Test error")
plt.show() 
            
#%%

with open("imdb_extrait3.pkl" ,"rb") as f:
    [data,id2titles, fields]=pickle.load(f)
datax = data[:,:32]
datay= 2*(data[:,33]>6.5)-1 # seuil de bon film a 6.5

#%%

iterations=500
step=1e-3
eps=1e-1

classifier=Perceptron(iterations)
classifier.fit(datax,datay,stochastic=False,w=None)

#classifier_sto=Perceptron(iterations)
#classifier_sto.fit(data,y,stochastic=True,w=None)
 
x=np.arange(iterations)
plt.figure()
plt.plot()
plt.plot(x,classifier.hist_f,"r")
#plt.plot(x,classifier_sto.hist_f,"b")
plt.title("Fonction loss avec pas={} et {} iterations".format(step,iterations))
plt.show()
