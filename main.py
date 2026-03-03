import data_cleaning

def run():
    print("=== Lancement du Nettoyage Complet (Projet SAAM) ===")
    
    # Étape 1 : Chargement
    df_raw = data_cleaning.charger_donnees_ri("DS_RI_T_USD_M_2025.xlsx")
    
    # Étape 2 : Nettoyage de base (Génère deux versions des données)
    df_avant_ffill, df_rempli = data_cleaning.nettoyer_donnees_financieres(df_raw)
    
    # Étape 3 : Calcul des rendements et gestion des entreprises délistées
    returns = data_cleaning.calculer_rendements_et_delistings(df_avant_ffill, df_rempli)
    
    # Étape 4 : Filtrage des prix figés (Stale prices)
    df_final, returns_final = data_cleaning.filtrer_stale_prices(df_rempli, returns, seuil=0.5)
    
    print("\n--- Aperçu des Rendements (Returns) après nettoyage ---")
    print(returns_final.head())
    
    # Étape 5 : Sauvegarde
    df_final.to_csv("data/cleaned/RI_final_clean.csv")
    returns_final.to_csv("data/cleaned/returns_final.csv")
    print("\nSuccès ! Fichiers sauvegardés dans 'data/cleaned/'.")

if __name__ == "__main__":
    run()