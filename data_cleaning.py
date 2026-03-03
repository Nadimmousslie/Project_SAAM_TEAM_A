import pandas as pd
import numpy as np
import os

def charger_donnees_ri(nom_fichier):
    chemin = os.path.join("data", nom_fichier)
    print(f"--- Chargement de : {nom_fichier} ---")
    return pd.read_excel(chemin, index_col=[0, 1])

def nettoyer_donnees_financieres(df):
    print("Début du nettoyage de base...")
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all')
    df_base = df.mask(df < 0.5)
    
    # On garde une copie AVANT de boucher les trous pour repérer la vraie fin de vie des entreprises
    df_avant_ffill = df_base.copy() 
    
    # On bouche les trous au milieu
    df_rempli = df_base.ffill(axis=1)
    return df_avant_ffill, df_rempli

def calculer_rendements_et_delistings(df_avant_ffill, df_rempli):
    print("Calcul des rendements et ajustement des faillites (-100%)...")
    # Calcul classique des rendements
    returns = df_rempli.transpose().pct_change().transpose()
    
    # 1. Identifier les entreprises délistées grâce à leur nom
    noms = returns.index.get_level_values('NAME')
    masque_deliste = noms.str.contains('DEAD|DELIST', case=False, na=False)
    firmes_mortes = returns[masque_deliste].index
    
    # 2. Ajuster les rendements pour chaque firme morte
    for firme in firmes_mortes:
        prix_originaux = df_avant_ffill.loc[firme]
        dernier_mois_valide = prix_originaux.last_valid_index()
        
        if dernier_mois_valide is not None:
            pos = returns.columns.get_loc(dernier_mois_valide)
            
            # Le mois SUIVANT le délisting = -100% de rendement (-1.0)
            if pos + 1 < len(returns.columns):
                mois_faillite = returns.columns[pos + 1]
                returns.loc[firme, mois_faillite] = -1.0  
                
                # Tous les mois APRÈS la faillite redeviennent des valeurs manquantes (NaN)
                if pos + 2 < len(returns.columns):
                    colonnes_suivantes = returns.columns[pos + 2:]
                    returns.loc[firme, colonnes_suivantes] = np.nan
                    
    return returns

def filtrer_stale_prices(df_rempli, returns, seuil=0.5):
    print(f"Filtrage des Stale Prices (exclusion si > {seuil*100}% de mois figés)...")
    
    # On compte la proportion de 0% (les NaN ne sont pas comptés dans la moyenne)
    proportion_zeros = (returns == 0).sum(axis=1) / returns.count(axis=1)
    firmes_valides = proportion_zeros <= seuil
    
    # On filtre les prix et les rendements
    df_filtre = df_rempli[firmes_valides]
    returns_filtre = returns[firmes_valides]
    
    nb_exclus = len(returns) - len(returns_filtre)
    print(f"Terminé. {nb_exclus} entreprises exclues car trop illiquides.")
    
    return df_filtre, returns_filtre