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
import multiprocessing
import pandas as pd
import numpy as np
import os
import pygal

os.chdir("/home/garance/Bureau/Delinquance/Data")

sheets=[x for x in range (3,104)]
dict_of_df=pd.read_excel("Tableaux_4001_TS.xlsx",sheet_name=sheets)

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
    print("done :",sheet-2)
array_travail=np.delete(array_travail,0,0)


"acquisition des données démographiques départements 2016"        
df=pd.read_excel("estim-pop-dep-sexe-gca-1975-2016.xls",sheet_name=1)
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

"préparation de la carte des données totales"
dict_delinquance_totale=df_Delinquance.sum(axis=1).to_dict()
carte_delinquance = pygal.maps.fr.Departments(human_readable=True, legend_at_bottom=True)
carte_delinquance.title = 'Délinquance totale par départements en 2016'
carte_delinquance.add('données 2016', dict_delinquance_totale)
carte_delinquance.render_in_browser()

"préparation de la carte des données totales"
dict_delinquance_ratio=df_Delinquance2.sum(axis=1).to_dict()
carte_delinquance_ratio = pygal.maps.fr.Departments(human_readable=True, legend_at_bottom=True)
carte_delinquance_ratio.title = 'Ratio Délinquance par départements en 2016'
carte_delinquance_ratio.add('données 2016', dict_delinquance_ratio)
carte_delinquance_ratio.render_in_browser()