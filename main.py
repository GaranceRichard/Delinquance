# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:18:39 2017

@author: garance
"""
import multiprocessing
import pandas as pd
import numpy as np

def import_departement(sheet):
    df=pd.read_excel("Data/Tableaux_4001_TS.xlsx",sheet_name=sheet)
    dft=df[["2016_01","2016_02","2016_03","2016_04","2016_05","2016_06","2016_07","2016_08","2016_09","2016_10","2016_11","2016_12"]]
    conv_arr= dft.values
    donnees = conv_arr.ravel()
    return donnees

if __name__ == '__main__':
    array_travail = import_departement(3)
    
    for sheet in range (4,104):
        array_transitoire=import_departement(sheet)
        array_travail=np.vstack((array_travail, array_transitoire))
        print("done :",sheet) 
        
    df=pd.read_excel("Data/estim-pop-dep-sexe-gca-1975-2016.xls",sheet_name=1)
    df=df.drop([0,1,2,3,100,106,107,108,109])
    df.columns=[x for x in range(20)]
    df=df[[0,1,13]]
    df=df.set_index([0])
    df.columns=["nom","population"]
        
    df2=pd.DataFrame(array_travail)
    df2.index=df.index
    df3=df2.astype('float').div(df['population'].astype('float'),axis='index')