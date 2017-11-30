# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:25:17 2017

@author: garance
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:16:29 2017

@author: garance
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:18:39 2017

@author: garance
"""
import pandas as pd
import numpy as np
import os
import pygal
import operator
import matplotlib.pyplot as plt
import time

from matplotlib import cm
from sklearn import manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from mpl_toolkits.mplot3d import Axes3D

os.chdir("/home/garance/Bureau/Delinquance")

"Récupération des données"
time0=time.clock()
sheets=[x for x in range (3,104)]
dict_of_df=pd.read_excel("Data/Tableaux_4001_TS.xlsx",sheet_name=sheets)

array_travail = np.empty(1284,)

"récupération des id comme titre de future colonnes"
final_columns=[]
for libelle in dict_of_df[4]["libellé index"].tolist():
    for t in range(1,13):
        final_columns.append(libelle+" "+str(t))

"mise en place de l'array de travail"
for sheet in dict_of_df.keys():
    df=dict_of_df[sheet][["2016_01","2016_02","2016_03","2016_04","2016_05","2016_06","2016_07","2016_08","2016_09","2016_10","2016_11","2016_12"]]
    donnees= df.values.ravel()
    array_travail=np.vstack((array_travail, donnees))
array_travail=np.delete(array_travail,0,0)
print("traitement des données délinquance :",time.clock()-time0)


"acquisition des données démographiques départements 2016"  
time1=time.clock()      
df=pd.read_excel("Data/estim-pop-dep-sexe-gca-1975-2016.xls",sheet_name=1)
df=df.drop([0,1,2,3,100,106,107,108,109])
df.columns=[x for x in range(20)]
df=df[[0,1,13]]
df=df.set_index([0])
df.columns=["nom","population"]
    
"array de travail devient DataFrame pandas"
df_Delinquance=pd.DataFrame(array_travail)
"récupération des index départementaux"
df_Delinquance.index=df.index
df_Delinquance.columns=final_columns
    
"les données brutes deviennent ratios en pour 1000"
df_Delinquance2=df_Delinquance.astype('float').div(df['population'].astype('float'),axis='index')
df_Delinquance2=df_Delinquance2*1000
print("traitement des données démographiques :",time.clock()-time1)


"Création des visuels des données"
time2=time.clock()
if not os.path.exists("Visuels"):
    os.makedirs("Visuels")
    
if not os.path.exists("Visuels/Données"):
    os.makedirs("Visuels/Données")

"préparation de la carte des données totales"
dict_delinquance_totale=df_Delinquance.sum(axis=1).to_dict()
carte_delinquance = pygal.maps.fr.Departments(human_readable=True, legend_at_bottom=True)
carte_delinquance.title = 'Délinquance totale par départements en 2016'
carte_delinquance.add('données 2016', dict_delinquance_totale)
carte_delinquance.render_to_file("Visuels/Données/carte_total.svg")

"préparation de la carte des données ratios"
dict_delinquance_ratio=df_Delinquance2.sum(axis=1).to_dict()
carte_delinquance_ratio = pygal.maps.fr.Departments(human_readable=True, legend_at_bottom=True)
carte_delinquance_ratio.title = 'Ratio Délinquance par départements en 2016'
carte_delinquance_ratio.add('données 2016', dict_delinquance_ratio)
carte_delinquance_ratio.value_formatter = lambda x: "%.2f" % x
carte_delinquance_ratio.render_to_file("Visuels/Données/carte_delinquance_ratio.svg")
print("Visualisation des données :",time.clock()-time2)

"comparaison des données"
time3=time.clock()
"préparation des données"

classement_valeur = [i[0] for i in sorted(dict_delinquance_totale.items(), key=operator.itemgetter(1), reverse=True)]
classement_ratio = [i[0] for i in sorted(dict_delinquance_ratio.items(), key=operator.itemgetter(1), reverse=True)]

tab=[]
value = 15

for val in classement_valeur[:value]:  
    tab.append([100-classement_valeur.index(val),100-classement_ratio.index(val),val])
for val in classement_ratio[:value]:
    if [100-classement_valeur.index(val),100-classement_ratio.index(val)] not in tab:
        tab.append([100-classement_valeur.index(val),100-classement_ratio.index(val),val])

"Visualisation des évolutions"
fig = plt.figure() 
ax = fig.add_subplot(1,1,1)
ax.axis("off")
   
for x in tab:
    if x[0]>x[1]:
        ax.plot([x[0],x[1]],'--o',color="red")
    elif x[0]==x[1]:
        ax.plot([x[0],x[1]],'--o',color="black")
    else:
        ax.plot([x[0],x[1]],'--o',color="green")
    if x[0]>100-value:
        ax.text(0-.1,x[0]-.1,x[2])
    if x[1]>100-value:
        ax.text(1.05,x[1]-.1,x[2])

ax.text(0-.1,101.1,"Données valeurs")
ax.text(1-.1,101.1,"Données ratios")
fig.suptitle('Valeurs versus ratio', fontsize=20)

ax.set_ylim([100-value-1,101])
ax.axvline(x=0)
ax.axvline(x=1)

fig.savefig('Visuels/Données/Valeurs_vs_Ratios.png')
plt.close()
print("Comparaison des données :",time.clock()-time3)


"Réduction dimensionnelle et Clusterization"
time4=time.clock()

if not os.path.exists("Visuels/Réduction"):
    os.makedirs("Visuels/Réduction")

if not os.path.exists("Visuels/Clusters"):
    os.makedirs("Visuels/Clusters")

"Variables types"
plt.set_cmap('Set1')
X=df_Delinquance2.values
y=df_Delinquance2.index
n_neighbors=10
level=3
np.seterr(divide='ignore', invalid='ignore')
silhouette_ref = 0

def plot_embedding(value,key):
    global choice, silhouette_ref, choicerep   
    time5=time.clock()
    val=[]
    lamda=[]    
    "standardisation"    
    x_min, x_max = np.min(value, 0), np.max(value, 0)
    reducer = (value - x_min) / (x_max - x_min)
    "clusterisation"    
    for repeat in range(10,14):
        clusterer = KMeans(n_clusters=repeat)
        cluster_labels = clusterer.fit_predict(reducer)
        score = silhouette_score(reducer,cluster_labels)
        X_projected=reducer
        title=str(key)+" for "+str(repeat)+" clusters, silhouette score = "+str(score)
        
        fig = plt.figure(figsize=(20,10))
        fig.suptitle(title, fontsize=15)        
        ax = fig.add_subplot(111, projection='3d')
        coulour=cluster_labels
        size = X_projected[:,0]**0+1000
        p=ax.scatter(X_projected[:,0],X_projected[:,1], X_projected[:,2],c=coulour, s=size)
        
        for i, (alpha,beta,gama) in enumerate(zip(X_projected[:,0], X_projected[:,1], X_projected[:,2])):
            ax.text(alpha,beta,gama, df.index[i],color='k',fontsize=15)
        
        fig.colorbar(p)

        chemin="Visuels/Clusters/"+str(key)+str(repeat)+".png"
        plt.savefig(chemin)
        plt.close()
        plot_map(X_projected,key,repeat)
        
        
        val.append(repeat)
        lamda.append(score)
        
            
    
    chemin = "Visuels/Réduction/"+str(key)+".png"
    plt.plot(val,lamda)
    plt.title(key)
    plt.savefig(chemin)
    plt.close()
    print("\t\t"+str(key)+" : "+str(time.clock()-time5))
    
    

def plot_map(value,key,nb_clust):
    nb_clust=nb_clust
    clusterer = KMeans(n_clusters=nb_clust)
    cluster_labels = clusterer.fit_predict(value)
    df_Cluster = pd.DataFrame(cluster_labels, columns=["Cluster"])
    df_Cluster.index=df_Delinquance2.index
    cluster_map = pygal.maps.fr.Departments()
    map_title= 'Carte clusterisée des départements par '+str(key)
    cluster_map.title = map_title
    for cluster in range(nb_clust):
        departements = df_Cluster[df_Cluster['Cluster'] == cluster].index.tolist()
        cluster_map.add("Cluster "+str(cluster+1),departements)
    cluster_map.render_in_browser()

"Différents modèles"
print("\t traitement des modèles :")

from sklearn import preprocessing
std_scale = preprocessing.StandardScaler().fit(X)
X = std_scale.transform(X)
   
rp = random_projection.SparseRandomProjection(n_components=level)
X_projected = rp.fit_transform(X)
plot_embedding(X_projected, "Random Projection")

X_pca = decomposition.TruncatedSVD(n_components=level).fit_transform(X)
plot_embedding(X_pca, "Projection en composentes principales")

X_iso = manifold.Isomap(n_neighbors, n_components=level).fit_transform(X)
plot_embedding(X_iso,"Projection Isomap")

clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=level, method='standard')
X_lle = clf.fit_transform(X)
plot_embedding(X_lle,"LLE" )

clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=level, method='modified')
X_mlle = clf.fit_transform(X)
plot_embedding(X_mlle,"LLE modifiée")

clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=level, method='hessian')
X_hlle = clf.fit_transform(X)
plot_embedding(X_hlle,"LLE Hessian")

clf = manifold.MDS(n_components=level, n_init=1, max_iter=100)
X_mds = clf.fit_transform(X)
plot_embedding(X_mds,"MDS")

hasher = ensemble.RandomTreesEmbedding(n_estimators=100)
X_transformed = hasher.fit_transform(X)
pca = decomposition.TruncatedSVD(n_components=level)
X_reduced = pca.fit_transform(X_transformed)
plot_embedding(X_reduced, "Random forest")

embedder = manifold.SpectralEmbedding(n_components=level, random_state=0, eigen_solver="arpack")
X_se = embedder.fit_transform(X)
plot_embedding(X_se, "Spectral embedding")

tsne = manifold.TSNE(n_components=level, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne, "t-SNE")

print("Réduction dimensionnelle et clustering :",time.clock()-time4)

