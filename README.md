# Project_SAAM_TEAM_A

# Pour Windows
# Création de l'environnement virtuel
python -m venv venv

# Activation de l'environnement
.\venv\Scripts\activate

# Mise à jour de pip et installation des bibliothèques
pip install --upgrade pip
pip install pandas numpy matplotlib scipy notebook cvxpy openpyxl

# Pour MAC

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install pandas numpy matplotlib scipy notebook cvxpy openpyxl 

# Débloquer l'activation de l'environnement viruel 
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\activate

# install pandas
pip install pandas openpyxl
