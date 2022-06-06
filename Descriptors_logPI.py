import rdkit
from rdkit import Chem 
from rdkit.Chem import Fragments
import pandas as pd
import scipy 
import sklearn
import numpy as np
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

### Funciones para contar grupos funcionales y generar un data frame con la informacion basica de las moleculas y los grupos
### funcionales contados

def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()
def mae(predictions, targets):
    return (np.abs(predictions - targets).mean())
def r2(predictions, targets):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(predictions, targets)
    return(r_value*r_value)
def regr_lin(predictions, targets):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(predictions, targets)
    print('slope: ', slope)
    print('intercept: ', intercept)
    print('p_value: ', p_value)
    print('std_err:, ', std_err)
def metricas(predictions, targets):
    print('R2: ', r2(predictions, targets))
    print('RMSE: ', rmse(predictions, targets))
    print('MSE: ', mse(predictions, targets))
    print('MAE: ', mae(predictions, targets))
    
def grupos_func(mol):
    functional_groups = [Fragments.fr_Al_COO(mol),
             Fragments.fr_Al_OH(mol),
             Fragments.fr_Al_OH_noTert(mol),
             Fragments.fr_ArN(mol),
             Fragments.fr_Ar_COO(mol),
             Fragments.fr_Ar_N(mol),
             Fragments.fr_Ar_NH(mol),
             Fragments.fr_Ar_OH(mol),
             Fragments.fr_COO(mol),
             Fragments.fr_COO2(mol),
             Fragments.fr_C_O(mol),
             Fragments.fr_C_O_noCOO(mol),
             Fragments.fr_C_S(mol),
             Fragments.fr_HOCCN(mol),
             Fragments.fr_Imine(mol),
             Fragments.fr_NH0(mol),
             Fragments.fr_NH1(mol),
             Fragments.fr_NH2(mol),
             Fragments.fr_N_O(mol),
             Fragments.fr_Ndealkylation1(mol),
             Fragments.fr_Ndealkylation2(mol),
             Fragments.fr_Nhpyrrole(mol),
             Fragments.fr_SH(mol),
             Fragments.fr_aldehyde(mol),
             Fragments.fr_alkyl_carbamate(mol),
             Fragments.fr_allylic_oxid(mol),
             Fragments.fr_amide(mol),
             Fragments.fr_amidine(mol),
             Fragments.fr_aniline(mol),
             Fragments.fr_aryl_methyl(mol),
             Fragments.fr_azide(mol),
             Fragments.fr_azo(mol),
             Fragments.fr_barbitur(mol),
             Fragments.fr_benzene(mol),
             Fragments.fr_benzodiazepine(mol),
             Fragments.fr_bicyclic(mol),
             Fragments.fr_diazo(mol),
             Fragments.fr_dihydropyridine(mol),
             Fragments.fr_epoxide(mol),
             Fragments.fr_ester(mol),
             Fragments.fr_ether(mol),
             Fragments.fr_furan(mol),
             Fragments.fr_guanido(mol),
             Fragments.fr_hdrzine(mol),
             Fragments.fr_hdrzone(mol),
             Fragments.fr_imidazole(mol),
             Fragments.fr_imide(mol),
             Fragments.fr_isocyan(mol),
             Fragments.fr_isothiocyan(mol),
             Fragments.fr_ketone(mol),
             Fragments.fr_ketone_Topliss(mol),
             Fragments.fr_lactam(mol),
             Fragments.fr_lactone(mol),
             Fragments.fr_methoxy(mol),
             Fragments.fr_morpholine(mol),
             Fragments.fr_nitrile(mol),
             Fragments.fr_nitro(mol),
             Fragments.fr_nitro_arom(mol),
             Fragments.fr_nitro_arom_nonortho(mol),
             Fragments.fr_nitroso(mol),
             Fragments.fr_oxazole(mol),
             Fragments.fr_oxime(mol),
             Fragments.fr_para_hydroxylation(mol),
             Fragments.fr_phenol(mol),
             Fragments.fr_phenol_noOrthoHbond(mol),
             Fragments.fr_phos_acid(mol),
             Fragments.fr_phos_ester(mol),
             Fragments.fr_piperdine(mol),
             Fragments.fr_piperzine(mol),
             Fragments.fr_priamide(mol),
             Fragments.fr_prisulfonamd(mol),
             Fragments.fr_pyridine(mol),
             Fragments.fr_quatN(mol),
             Fragments.fr_sulfide(mol),
             Fragments.fr_sulfonamd(mol),
             Fragments.fr_sulfone(mol),
             Fragments.fr_term_acetylene(mol),
             Fragments.fr_tetrazole(mol),
             Fragments.fr_thiazole(mol),
             Fragments.fr_thiocyan(mol),
             Fragments.fr_thiophene(mol),
             Fragments.fr_unbrch_alkane(mol),
             Fragments.fr_urea(mol)]
    columns=['fr_Al_COO','fr_Al_OH','fr_Al_OH_noTert','fr_ArN','fr_Ar_COO','fr_Ar_N','fr_Ar_NH','fr_Ar_OH','fr_COO',
                'fr_COO2','fr_C_O','fr_C_O_noCOO','fr_C_S','fr_HOCCN','fr_Imine','fr_NH0','fr_NH1','fr_NH2','fr_N_O',
                'fr_Ndealkylation1','fr_Ndealkylation2','fr_Nhpyrrole','fr_SH','fr_aldehyde','fr_alkyl_carbamate','fr_allylic_oxid',
                'fr_amide','fr_amidine','fr_aniline','fr_aryl_methy','fr_azide',
                'fr_azo','fr_barbitur','fr_benzene','fr_benzodiazepine','fr_bicyclic','fr_diazo','fr_dihydropyridine',
                'fr_epoxide','fr_ester','fr_ether','fr_furan','fr_guanido','fr_hdrzine','fr_hdrzone',
                'fr_imidazole','fr_imide','fr_isocyan','fr_isothiocyan','fr_ketone','fr_ketone_Topliss','fr_lactam',
                'fr_lactone','fr_methoxy','fr_morpholine','fr_nitrile','fr_nitro','fr_nitro_arom','fr_nitro_arom_nonortho',
                'fr_nitroso','fr_oxazole','fr_oxime','fr_para_hydroxylation','fr_phenol','fr_phenol_noOrthoHbond',
                'fr_phos_acid','fr_phos_ester','fr_piperdine','fr_piperzine','fr_priamide','fr_prisulfonamd','fr_pyridine',
                'fr_quatN','fr_sulfide','fr_sulfonamd','fr_sulfone','fr_term_acetylene','fr_tetrazole','fr_thiazole',
                'fr_thiocyan','fr_thiophen','fr_unbrch_alkane','fr_urea']
    functional_groups = np.array(functional_groups)
    f = pd.DataFrame(np.reshape(functional_groups, (1, len(columns))), columns = columns)
    return f

def descr(mol):
    Carbono = Chem.MolFromSmarts("[#6]")
    Cloro = Chem.MolFromSmarts("[#17]")
    Fluor = Chem.MolFromSmarts("[#9]")
    Bromo = Chem.MolFromSmarts("[#35]")
    Yodo = Chem.MolFromSmarts("[#53]")
    descriptores = [rdkit.Chem.Descriptors.MolWt(mol),
                       rdkit.Chem.Descriptors.MolMR(mol),
                       rdkit.Chem.Descriptors.NumHAcceptors(mol),
                       rdkit.Chem.Descriptors.NumHDonors(mol),
                       rdkit.Chem.Descriptors.NumRotatableBonds(mol),
                       rdkit.Chem.Descriptors.NumAromaticRings(mol),
                       rdkit.Chem.Descriptors.NumAliphaticRings(mol),
                       rdkit.Chem.Descriptors.TPSA(mol),
                       len(mol.GetSubstructMatches(Carbono)),
                       len(mol.GetSubstructMatches(Cloro)),
                       len(mol.GetSubstructMatches(Fluor)),
                       len(mol.GetSubstructMatches(Bromo)),
                       len(mol.GetSubstructMatches(Yodo))
                       ]
    columns=['MW','MR','HBA','HBD','RotBonds','AromRings','AliphRings','PSA','C','Cl','F','Br','I']
    descriptores = np.array(descriptores)
    f = pd.DataFrame(np.reshape(descriptores, (1, len(columns))), columns = columns)
    return f

def descr_carg(mol):
    descriptores = [rdkit.Chem.Descriptors.MolMR(mol),
                    rdkit.Chem.Descriptors.TPSA(mol)]
    columns=['MR_carg','PSA_carg']
    descriptores = np.array(descriptores)
    f = pd.DataFrame(np.reshape(descriptores, (1, 2)), columns=columns)
    return f
        
def conteo_descr_(mol1, mol2): 
    Carga = pd.DataFrame(np.reshape(Chem.rdmolops.GetFormalCharge(mol2), (1, 1)), columns = ['Carga'])
    Grupos = grupos_func(mol1)
    Descriptores = descr(mol1)
    Des_carg = descr_carg(mol2)
    dflogP = pd.concat([Carga, Grupos, Descriptores, Des_carg], axis=1)
    return(dflogP)

##### Funcion grafico 

def scatter(pred, target, model, graph_name, anot):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pred, target)
    plt.figure(figsize=(8,6))
    col = np.where(np.abs(pred - target) < 3*rmse(pred, target),'#4d66c9', 'orange')
    plt.scatter(pred, target, s=30, color = col)
    plt.plot([np.min(target), np.max(target)], [np.min(target), np.max(target)], color = "gray")
    plt.xlabel("Delta_" + model)
    plt.ylabel("Delta_exp")
    z = np.polyfit(pred, target, 1)
    p = np.poly1d(z)
    plt.plot(pred, p(pred), "--", color = "black")
    

    big_error = []
    for i in range(len(target)):
        if np.abs(pred[i] - target[i]) > 3*rmse(pred, target):
            big_error.append(i)
    annotations = np.array(target.index)[big_error]
    New_Y_anno = np.array(pred)[big_error]
    Act_Y_anno = np.array(target)[big_error]
    
    if anot == True:
        for i in range(len(annotations)):
            plt.annotate(annotations[i], (New_Y_anno[i] + 0.1, Act_Y_anno[i]), fontsize = 9)
    
    
    if intercept < 0: sign = ' - '
    else: sign = ' + '
    
    intercept = np.abs(intercept)
    
    plt.annotate('$R^2$: ' + str(np.round(r_value*r_value, decimals = 2)), xy=(0.10, 0.90), xycoords='axes fraction',
                fontsize = 12)
    plt.annotate('RMSE: ' + str(np.round(rmse(pred, target), decimals = 2)), xy=(0.10, 0.85), xycoords='axes fraction',
                fontsize = 12)
    plt.annotate('MSE: ' + str(np.round(mse(pred, target), decimals = 2)), xy=(0.10, 0.80), xycoords='axes fraction',
                fontsize = 12)
    plt.annotate('MAE: ' + str(np.round(mae(pred, target), decimals = 2)), xy=(0.10, 0.75), xycoords='axes fraction',
                fontsize = 12)
    plt.annotate('y = ' + str(np.round(slope, decimals = 2)) + 'x' + sign + str(np.round(intercept, decimals = 2)), xy=(0.10, 0.65), 
                xycoords='axes fraction',
                fontsize = 12)
    count = 0
    for i in col: 
        if i == 'orange': count += 1
    print('Puntos fuera: ', count)
    
    plt.savefig(graph_name, dpi=500)