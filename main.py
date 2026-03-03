import data_cleaning
import os

def run():
    print("=== Lancement du Nettoyage Complet (Projet SAAM) ===")
    
    # Sécurité : on s'assure que le dossier 'cleaned' existe bien
    os.makedirs("data/cleaned", exist_ok=True)
    
    # --- ETAPE 1 : PRIX (RI) ---
    print("\n>>> ETAPE 1 : Nettoyage des prix (RI) et Rendements")
    df_raw_ri = data_cleaning.charger_donnees("DS_RI_T_USD_M_2025.xlsx")
    df_avant_ffill, df_rempli = data_cleaning.nettoyer_donnees_financieres(df_raw_ri)
    returns = data_cleaning.calculer_rendements_et_delistings(df_avant_ffill, df_rempli)
    df_ri_final, returns_final = data_cleaning.filtrer_stale_prices(df_rempli, returns, seuil=0.5)
    
    df_ri_final.to_csv("data/cleaned/DS_RI_T_USD_M_2025_clean.csv")
    returns_final.to_csv("data/cleaned/returns_final.csv")
    print("-> Fichiers RI et returns sauvegardés.")
    
    # --- ETAPE 2 : AUTRES SERIES TEMPORELLES (MV, REV, CO2) ---
    print("\n>>> ETAPE 2 : Nettoyage de la Market Value, Revenus et CO2")
    fichiers_series = [
        "DS_MV_T_USD_M_2025.xlsx",
        "DS_MV_T_USD_Y_2025.xlsx",
        "DS_REV_Y_2025.xlsx",
        "DS_CO2_SCOPE_1_Y_2025.xlsx"
    ]
    
    for fichier in fichiers_series:
        df_raw = data_cleaning.charger_donnees(fichier)
        df_clean = data_cleaning.nettoyer_donnees_generiques(df_raw)
        
        nom_sortie = fichier.replace(".xlsx", "_clean.csv")
        df_clean.to_csv(f"data/cleaned/{nom_sortie}")
        print(f"-> Sauvegardé sous : {nom_sortie}")

    # --- ETAPE 3 : FICHIER STATIQUE ---
    print("\n>>> ETAPE 3 : Nettoyage du fichier statique")
    # index_col=None car c'est un tableau normal (pas un tableau avec dates)
    df_static_raw = data_cleaning.charger_donnees("Static_2025.xlsx", index_col=None)
    df_static_clean = data_cleaning.nettoyer_donnees_statiques(df_static_raw)
    
    df_static_clean.to_csv("data/cleaned/Static_2025_clean.csv", index=False)
    print("-> Sauvegardé sous : Static_2025_clean.csv")

    print("\n=== ✨ TOUT EST NETTOYÉ AVEC SUCCÈS ! ✨ ===")

if __name__ == "__main__":
    run()