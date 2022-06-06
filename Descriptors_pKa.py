import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors

def descriptor(mol1, mol2, Type):
    p1 = pd.DataFrame([mol1], columns = ['Mol'])
    for i,j in Descriptors.descList:
        p1[i] = p1.Mol.map(j)

    p2 = pd.DataFrame([mol2], columns = ['Mol'])
    for i,j in Descriptors.descList:
        p2[i] = p2.Mol.map(j)

    
    p1 = p1.drop('Mol', axis = 1)
    p2 = p2.drop('Mol', axis = 1)
    
    p = p2 - p1
    
    if Type == 'acid':
        p.loc[i, 'Tipo'] = -1
    elif Type == 'basic':
        p.loc[i, 'Tipo'] =  1
    
    train = pd.read_csv('train_pKa.csv')
    
    p = p[train.columns[5:]]
    
    return p 