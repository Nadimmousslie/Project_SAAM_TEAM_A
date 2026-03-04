import pandas as pd
import numpy as np
import os

# ==========================================
# 1. FONCTIONS DE CHARGEMENT
# ==========================================
def charger_donnees(nom_fichier, index_col=[0, 1]):
    chemin = os.path.join("data", nom_fichier)
    print(f"--- Chargement de : {nom_fichier} ---")
    return pd.read_excel(chemin, index_col=index_col)

# ==========================================
# 2. FONCTIONS POUR LES PRIX (RI) UNIQUEMENT
# ==========================================
def nettoyer_donnees_financieres(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all')
    df_base = df.mask(df < 0.5) # Low prices
    
    df_avant_ffill = df_base.copy()
    df_rempli = df_base.ffill(axis=1) # Missing values (middle)
    return df_avant_ffill, df_rempli

def calculer_rendements_et_delistings(df_avant_ffill, df_rempli):
    returns = df_rempli.transpose().pct_change().transpose()
    
    noms = returns.index.get_level_values('NAME')
    masque_deliste = noms.str.contains('DEAD|DELIST', case=False, na=False)
    firmes_mortes = returns[masque_deliste].index
    
    for firme in firmes_mortes:
        prix_originaux = df_avant_ffill.loc[firme]
        dernier_mois_valide = prix_originaux.last_valid_index()
        
        if dernier_mois_valide is not None:
            pos = returns.columns.get_loc(dernier_mois_valide)
            if pos + 1 < len(returns.columns):
                mois_faillite = returns.columns[pos + 1]
                returns.loc[firme, mois_faillite] = -1.0  # -100% rendement faillite
                
                if pos + 2 < len(returns.columns):
                    colonnes_suivantes = returns.columns[pos + 2:]
                    returns.loc[firme, colonnes_suivantes] = np.nan
                    
    return returns

def filtrer_stale_prices(df_rempli, returns, seuil=0.5):
    proportion_zeros = (returns == 0).sum(axis=1) / returns.count(axis=1)
    firmes_valides = proportion_zeros <= seuil
    return df_rempli[firmes_valides], returns[firmes_valides]

# ==========================================
# 3. FONCTION POUR MV, REV et CO2
# ==========================================
def nettoyer_donnees_generiques(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all')
    df = df.ffill(axis=1) # Remplir les trous avec la valeur précédente
    return df

# ==========================================
# 4. FONCTION POUR LE FICHIER STATIC
# ==========================================
def nettoyer_donnees_statiques(df):
    # Pas de ffill ici, juste supprimer les lignes totalement vides
    df = df.dropna(how='all')
    return df