# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:18:39 2017

@author: garance
"""
import multiprocessing
import pandas as pd
import numpy as np

def import_departement(sheet):
    "fonction qui importe les départements"
    df=pd.read_excel("Data/Tableaux_4001_TS.xlsx",sheet_name=sheet)
    "on ne conserve que l'année 2016"
    df=df[["2016_01","2016_02","2016_03","2016_04","2016_05","2016_06","2016_07","2016_08","2016_09","2016_10","2016_11","2016_12"]]
    "on reshape sur une ligne"
    donnees= df.values.ravel()
    return donnees

if __name__ == '__main__':
    "initialisation de l'array"    
    array_travail = np.empty(1284,)
    
    "récupération des départements 01 à 976"
    for sheet in range (3,104):
        array_transitoire=import_departement(sheet)
        array_travail=np.vstack((array_travail, array_transitoire))
        print("done :",sheet-2)
        
    "acquisition des données démographiques"        
    df=pd.read_excel("Data/estim-pop-dep-sexe-gca-1975-2016.xls",sheet_name=1)
    
    "Wangling des données démographiques"
    df=df.drop([0,1,2,3,100,106,107,108,109])
    df.columns=[x for x in range(20)]
    df=df[[0,1,13]]
    df=df.set_index([0])
    df.columns=["nom","population"]
    
    "array de travail devient DataFrame pandas"
    df_Delinquance=pd.DataFrame(array_travail)
    "récupération des index départementaux"
    df_Delinquance.index=df.index
    
    "les données brutes deviennent ratios 1/1"
    df_Delinquance=df_Delinquance.astype('float').div(df['population'].astype('float'),axis='index')