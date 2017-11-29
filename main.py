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
print("Comparaison des données :",time.clock()-time3)


