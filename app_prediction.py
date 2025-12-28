import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Prix Voiture",
    page_icon="🚗",
    layout="wide"
)

# Titre principal
st.title("🚗 Prédiction du Prix des Voitures")
st.markdown("---")

# Charger le modèle et les objets de preprocessing
@st.cache_resource
def load_model_and_preprocessors():
    """Charge le modèle et tous les objets de preprocessing"""
    try:
        model = joblib.load('XGBoost_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        preprocessing_info = joblib.load('preprocessing_info.pkl')
        return model, scaler, label_encoders, preprocessing_info
    except FileNotFoundError as e:
        st.error(f"❌ Erreur: Fichier manquant - {e}")
        st.info("Veuillez d'abord exécuter le notebook modeling.ipynb pour sauvegarder le modèle.")
        return None, None, None, None

@st.cache_data
def load_marque_modele_mapping():
    """Charge le mapping marque-modèle depuis le fichier CSV"""
    try:
        # Essayer de charger le fichier nettoyé en premier
        try:
            df = pd.read_csv('tayara_cars_cleaned.csv')
        except:
            df = pd.read_csv('tayara_cars_all_pages.csv')
        
        # Créer un dictionnaire marque -> liste de modèles
        marque_modele_dict = {}
        df_valid = df[(df['marque'].notna()) & (df['modele'].notna())]
        
        for marque in df_valid['marque'].unique():
            modeles = df_valid[df_valid['marque'] == marque]['modele'].unique()
            marque_modele_dict[marque] = sorted(list(modeles))
        
        return marque_modele_dict
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger le mapping marque-modèle: {e}")
        return None

# Charger les objets
model, scaler, label_encoders, preprocessing_info = load_model_and_preprocessors()
marque_modele_dict = load_marque_modele_mapping()

if model is not None:
    # Afficher les informations du modèle
    with st.expander("ℹ️ Informations sur le modèle", expanded=False):
        st.write(f"**Modèle**: Random Forest Regressor")
        st.write(f"**Nombre d'arbres**: {model.n_estimators}")
        st.write(f"**Profondeur maximale**: {model.max_depth}")
        st.write(f"**Variables**: {len(preprocessing_info['all_features'])} features")
    
    st.markdown("### 📝 Entrez les caractéristiques du véhicule")
    
    # Créer deux colonnes pour l'interface
    col1, col2 = st.columns(2)
    
    # Dictionnaire pour stocker les valeurs saisies
    input_data = {}
    
    with col1:
        st.markdown("#### 🔢 Caractéristiques Numériques")
        
        # Année
        input_data['annee'] = st.number_input(
            "Année de fabrication",
            min_value=1990,
            max_value=2025,
            value=2015,
            step=1
        )
        
        # Cylindrée
        input_data['cylindree'] = st.number_input(
            "Cylindrée (cm³)",
            min_value=500,
            max_value=8000,
            value=1600,
            step=100
        )
        
        # Puissance fiscale
        input_data['puissance_fiscale'] = st.number_input(
            "Puissance fiscale (CV)",
            min_value=1,
            max_value=30,
            value=7,
            step=1
        )
        
        # Kilométrage
        kilometrage = st.number_input(
            "Kilométrage (km)",
            min_value=0,
            max_value=500000,
            value=80000,
            step=1000
        )
        # Stocker à la fois le kilométrage brut et sa transformation log
        input_data['kilometrage'] = kilometrage
        input_data['kilometrage_log'] = np.log1p(kilometrage)
    
    with col2:
        st.markdown("#### 📋 Caractéristiques Catégoriques")
        
        # Couleur du véhicule
        if 'couleur_du_vehicule' in label_encoders:
            couleur_options = list(label_encoders['couleur_du_vehicule'].classes_)
            input_data['couleur_du_vehicule'] = st.selectbox(
                "Couleur du véhicule",
                couleur_options
            )
        
        # État du véhicule
        if 'etat_du_vehicule' in label_encoders:
            etat_options = list(label_encoders['etat_du_vehicule'].classes_)
            input_data['etat_du_vehicule'] = st.selectbox(
                "État du véhicule",
                etat_options
            )
        
        # Boîte de vitesse
        if 'boite' in label_encoders:
            boite_options = list(label_encoders['boite'].classes_)
            input_data['boite'] = st.selectbox(
                "Boîte de vitesse",
                boite_options
            )
        
        # Marque
        if 'marque' in label_encoders:
            marque_options = list(label_encoders['marque'].classes_)
            input_data['marque'] = st.selectbox(
                "Marque",
                marque_options,
                index=0
            )
        
        # Modèle - Filtré par marque si mapping disponible
        if 'modele' in label_encoders:
            if marque_modele_dict and input_data.get('marque') in marque_modele_dict:
                # Filtrer les modèles selon la marque sélectionnée
                modele_options = marque_modele_dict[input_data['marque']]
                # S'assurer que les modèles sont dans les classes du label encoder
                modele_options = [m for m in modele_options if m in label_encoders['modele'].classes_]
            else:
                # Fallback: tous les modèles
                modele_options = list(label_encoders['modele'].classes_)
            
            input_data['modele'] = st.selectbox(
                "Modèle",
                modele_options,
                index=0 if len(modele_options) > 0 else 0
            )
        
        # Type de carrosserie
        if 'type_de_carrosserie' in label_encoders:
            carrosserie_options = list(label_encoders['type_de_carrosserie'].classes_)
            input_data['type_de_carrosserie'] = st.selectbox(
                "Type de carrosserie",
                carrosserie_options
            )
        
        # Carburant
        if 'carburant' in label_encoders:
            carburant_options = list(label_encoders['carburant'].classes_)
            input_data['carburant'] = st.selectbox(
                "Carburant",
                carburant_options
            )
    
    st.markdown("---")
    
    # Bouton de prédiction
    if st.button("🔮 Prédire le Prix", type="primary", use_container_width=True):
        try:
            # Créer un DataFrame avec les données saisies
            df_input = pd.DataFrame([input_data])
            
            # Appliquer le LabelEncoder sur les variables catégoriques
            categorical_cols = preprocessing_info['categorical_cols']
            for col in categorical_cols:
                if col in df_input.columns and col in label_encoders:
                    # Vérifier si la valeur existe dans les classes
                    if df_input[col].iloc[0] in label_encoders[col].classes_:
                        df_input[col] = label_encoders[col].transform(df_input[col])
                    else:
                        st.error(f"❌ Valeur '{df_input[col].iloc[0]}' non reconnue pour {col}")
                        st.stop()
            
            # Réorganiser les colonnes dans le bon ordre
            df_input = df_input[preprocessing_info['all_features']]
            
            # Appliquer le StandardScaler sur les variables numériques
            numerical_cols = preprocessing_info['numerical_cols']
            df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])
            
            # Faire la prédiction (sur l'échelle log)
            prediction_log = model.predict(df_input)[0]
            
            # Convertir la prédiction en prix réel (inverse de log transformation)
            prediction_prix = np.expm1(prediction_log)
            
            # Afficher le résultat
            st.markdown("---")
            st.markdown("### 🎯 Résultat de la Prédiction")
            
            # Créer 3 colonnes pour afficher les résultats
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric(
                    label="Prix Prédit (TND)",
                    value=f"{prediction_prix:,.0f}",
                    delta=None
                )
            
            with res_col2:
                st.metric(
                    label="Prix Prédit (Log)",
                    value=f"{prediction_log:.2f}",
                    delta=None
                )
            
            with res_col3:
                # Calculer un intervalle de confiance approximatif (±15%)
                lower_bound = prediction_prix * 0.85
                upper_bound = prediction_prix * 1.15
                st.metric(
                    label="Intervalle estimé (±15%)",
                    value=f"{lower_bound:,.0f} - {upper_bound:,.0f}",
                    delta=None
                )
            
            # Afficher les détails de la prédiction
            with st.expander("📊 Détails de la prédiction", expanded=True):
                st.markdown("**Résumé des caractéristiques:**")
                
                details_col1, details_col2 = st.columns(2)
                
                with details_col1:
                    st.write(f"- **Marque**: {input_data['marque']}")
                    st.write(f"- **Modèle**: {input_data['modele']}")
                    st.write(f"- **Année**: {input_data['annee']}")
                    st.write(f"- **Kilométrage**: {kilometrage:,} km")
                    st.write(f"- **Cylindrée**: {input_data['cylindree']} cm³")
                
                with details_col2:
                    st.write(f"- **Puissance fiscale**: {input_data['puissance_fiscale']} CV")
                    st.write(f"- **Carburant**: {input_data['carburant']}")
                    st.write(f"- **Boîte**: {input_data['boite']}")
                    st.write(f"- **Carrosserie**: {input_data['type_de_carrosserie']}")
                    st.write(f"- **État**: {input_data['etat_du_vehicule']}")
                    st.write(f"- **Couleur**: {input_data['couleur_du_vehicule']}")
            
            st.success("✅ Prédiction effectuée avec succès!")
            
            # Avertissement
            st.info("ℹ️ Cette prédiction est basée sur un modèle Random Forest entraîné sur des données historiques. Le prix réel peut varier en fonction des conditions du marché.")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            st.exception(e)
    
    # Section d'information supplémentaire
    st.markdown("---")
    with st.expander("❓ Comment utiliser cette application", expanded=False):
        st.markdown("""
        ### Guide d'utilisation
        
        1. **Remplissez les caractéristiques numériques** dans la colonne de gauche:
           - Année de fabrication
           - Cylindrée
           - Puissance fiscale
           - Kilométrage
        
        2. **Sélectionnez les caractéristiques catégoriques** dans la colonne de droite:
           - Couleur, État, Boîte de vitesse
           - Marque, Modèle
           - Type de carrosserie, Carburant
        
        3. **Cliquez sur "Prédire le Prix"** pour obtenir l'estimation
        
        4. **Consultez le résultat**:
           - Prix prédit en Dinars Tunisiens (TND)
           - Valeur log-transformée
           - Intervalle de confiance estimé
        
        ### Notes importantes
        - Le modèle a été entraîné sur des données du marché tunisien
        - La prédiction est basée sur un modèle Random Forest
        - Un intervalle de ±15% est fourni comme indication de variabilité
        """)

else:
    st.error("❌ Impossible de charger le modèle. Veuillez exécuter le notebook modeling.ipynb d'abord.")
    st.info("📌 Assurez-vous que les fichiers suivants existent:\n- random_forest_model.pkl\n- scaler.pkl\n- label_encoders.pkl\n- preprocessing_info.pkl")
