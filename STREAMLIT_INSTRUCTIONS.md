# 🚗 Application Streamlit - Prédiction Prix Voiture

## 📋 Prérequis

Avant de lancer l'application, assurez-vous d'avoir:

1. **Exécuté le notebook `modeling.ipynb`** jusqu'à la dernière cellule pour sauvegarder:
   - `random_forest_model.pkl` (modèle Random Forest)
   - `scaler.pkl` (StandardScaler)
   - `label_encoders.pkl` (LabelEncoders pour variables catégoriques)
   - `preprocessing_info.pkl` (informations sur les colonnes)

2. **Installé Streamlit**:
   ```bash
   pip install streamlit
   ```

## 🚀 Lancer l'application

Dans le terminal, depuis le dossier du projet, exécutez:

```bash
streamlit run app_prediction.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse: `http://localhost:8501`

## 📱 Utilisation de l'application

### Interface utilisateur

L'application est divisée en deux colonnes:

#### Colonne de gauche - Caractéristiques Numériques 🔢
- **Année de fabrication**: Entre 1990 et 2025
- **Cylindrée**: En cm³ (500 à 8000)
- **Puissance fiscale**: En CV (1 à 30)
- **Kilométrage**: En km (0 à 500,000)

#### Colonne de droite - Caractéristiques Catégoriques 📋
- **Couleur du véhicule**
- **État du véhicule**
- **Boîte de vitesse**
- **Marque**
- **Modèle**
- **Type de carrosserie**
- **Carburant**

### Faire une prédiction

1. Remplissez tous les champs avec les caractéristiques du véhicule
2. Cliquez sur le bouton **"🔮 Prédire le Prix"**
3. Consultez le résultat:
   - **Prix prédit en TND** (Dinars Tunisiens)
   - **Prix log-transformé** (valeur technique)
   - **Intervalle de confiance** (±15%)

## 🔧 Preprocessing appliqué

L'application applique automatiquement le même preprocessing que lors de l'entraînement:

1. **Transformation logarithmique** du kilométrage: `log1p(kilometrage)`
2. **Label Encoding** des variables catégoriques (7 variables)
3. **Standardisation** des variables numériques avec StandardScaler

## 📊 Informations sur le modèle

- **Modèle**: Random Forest Regressor
- **Nombre d'arbres**: 100
- **Profondeur maximale**: 15
- **Variables**: 11 features au total

### Performances du modèle

Les performances peuvent être consultées dans le notebook `modeling.ipynb`:
- Test R²
- Test RMSE
- Test MAE
- Cross-validation RMSE

## ⚠️ Notes importantes

- Les prédictions sont basées sur des données historiques du marché tunisien
- L'intervalle de confiance (±15%) est une estimation approximative
- Le prix réel peut varier selon les conditions du marché
- Assurez-vous que toutes les valeurs saisies sont cohérentes

## 🛠️ Dépannage

### Erreur "Fichier manquant"
➡️ Exécutez d'abord la dernière cellule du notebook `modeling.ipynb`

### Erreur "Module not found: streamlit"
➡️ Installez streamlit: `pip install streamlit`

### L'application ne se lance pas
➡️ Vérifiez que vous êtes dans le bon dossier et que `app_prediction.py` existe

### Valeur non reconnue pour une variable
➡️ Assurez-vous d'utiliser les valeurs disponibles dans les listes déroulantes

## 📁 Structure des fichiers

```
Prediction_prix_voiture-/
├── app_prediction.py              # Application Streamlit
├── modeling.ipynb                 # Notebook d'entraînement
├── PreProcessing.ipynb            # Notebook de preprocessing
├── tayara_cars_cleaned.csv        # Dataset nettoyé
├── random_forest_model.pkl        # Modèle sauvegardé
├── scaler.pkl                     # StandardScaler sauvegardé
├── label_encoders.pkl             # LabelEncoders sauvegardés
├── preprocessing_info.pkl         # Informations de preprocessing
└── STREAMLIT_INSTRUCTIONS.md      # Ce fichier
```

## 💡 Améliorations futures possibles

- Ajouter d'autres modèles (XGBoost, etc.)
- Afficher l'importance des features
- Permettre le téléchargement des prédictions
- Ajouter des graphiques comparatifs
- Interface multilingue (FR/AR/EN)
