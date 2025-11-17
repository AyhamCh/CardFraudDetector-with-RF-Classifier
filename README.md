# 🛡️ Système de Détection de Fraude par Machine Learning

## 📋 Description du Projet

Ce projet implémente un système de détection de fraude bancaire utilisant des techniques de Machine Learning. Il traite un dataset déséquilibré de transactions financières et utilise des méthodes d'undersampling et de SMOTE pour créer un modèle de classification efficace.

## 🎯 Objectifs

- Détecter les transactions frauduleuses dans un dataset bancaire
- Gérer le déséquilibre extrême des classes (fraude vs non-fraude)
- Éviter l'overfitting et le data leakage
- Obtenir un modèle performant et généralisable

## 📊 Dataset

**Fichier source :** `fraudTrain.csv`

**Caractéristiques :**
- Nombre initial de transactions : ~1,296,675
- Distribution initiale : ~0.58% de fraudes (hautement déséquilibré)
- Features : 23 colonnes incluant montants, catégories, informations géographiques, etc.

### Colonnes principales :
- `is_fraud` : Variable cible (0 = légitime, 1 = fraude)
- `amt` : Montant de la transaction
- `category` : Catégorie de dépense
- `lat`, `long` : Coordonnées géographiques
- Features démographiques et temporelles

## 🔧 Technologies Utilisées

```python
- Python 3.x
- pandas : Manipulation de données
- numpy : Calculs numériques
- scikit-learn : Modèles ML et métriques
- imbalanced-learn : Gestion du déséquilibre (SMOTE)
- matplotlib/seaborn : Visualisation (optionnel)
```

## 📦 Installation

```bash
# Cloner le repository
git clone <votre-repo>
cd fraud-detection

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

## 🚀 Utilisation

### 1. Préparation des données

```python
python prepare_data.py
```

Cette étape :
- Charge le dataset `fraudTrain.csv`
- Applique l'undersampling (réduction de 1,100,000 transactions non-frauduleuses)
- Mélange aléatoirement les données

### 2. Entraînement du modèle

```python
python train_model.py
```

Pipeline complet :
1. **Undersampling** : Réduction à 50,000 transactions non-frauduleuses
2. **Split Train/Test** : 80/20 stratifié
3. **SMOTE** : Oversampling de la classe minoritaire (uniquement sur train)
4. **Entraînement** : Random Forest avec hyperparamètres optimisés
5. **Évaluation** : Métriques détaillées sur le test set

### 3. Évaluation et prédiction

```python
python evaluate_model.py
```

## 📈 Résultats Attendus

### Distribution des classes après traitement :

```
Avant undersampling :
- Classe 0 (non-fraude) : 1,289,169
- Classe 1 (fraude) : 7,506

Après undersampling :
- Classe 0 : 50,000
- Classe 1 : 7,506

Après SMOTE (train uniquement) :
- Classe 0 : ~40,000
- Classe 1 : ~32,000 (ratio 0.8)
```

### Métriques de performance :

- **Accuracy** : 85-95% (sur test set)
- **Precision** : Minimiser les faux positifs
- **Recall** : Maximiser la détection des vraies fraudes
- **F1-Score** : Équilibre entre precision et recall

## ⚠️ Problèmes Courants et Solutions

### 1. Accuracy = 100% (Overfitting)

**Causes :**
- Data leakage (colonnes qui révèlent la target)
- SMOTE appliqué avant le split train/test
- Colonnes ID ou timestamps incluses

**Solutions :**
```python
# Exclure les colonnes suspectes
colonnes_a_exclure = ['trans_num', 'unix_time', 'trans_date_trans_time']

# SMOTE APRÈS le split
X_train, X_test = train_test_split(...)
X_train_resampled = smote.fit_resample(X_train)  # Seulement sur train !
```

### 2. Classes non mélangées

**Problème :** Les 0 et 1 sont groupés ensemble

**Solution :**
```python
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
```

### 3. Mémoire insuffisante

**Solution :** Augmenter l'undersampling initial
```python
sampled_non_fraud = non_fraud_df.sample(n=30000, random_state=42)  # Réduire à 30k
```

## 📁 Structure du Projet

```
fraud-detection/
│
├── fraudTrain.csv              # Dataset brut
├── prepare_data.py             # Script de préparation
├── train_model.py              # Script d'entraînement
├── evaluate_model.py           # Script d'évaluation
├── fraud_correct_pipeline.py   # Pipeline complet
├── requirements.txt            # Dépendances Python
├── README.md                   # Ce fichier
│
├── models/                     # Modèles sauvegardés
│   └── fraud_detector.pkl
│
├── results/                    # Résultats et visualisations
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── metrics_report.txt
│
└── notebooks/                  # Notebooks d'exploration
    └── exploration.ipynb
```

## 🔍 Bonnes Pratiques Implémentées

1. ✅ **Split avant SMOTE** : Évite le data leakage
2. ✅ **Stratified split** : Maintient la distribution des classes
3. ✅ **Mélange aléatoire** : Évite les patterns liés à l'ordre
4. ✅ **Exclusion des colonnes suspectes** : Prévient le data leakage
5. ✅ **Limitation de la profondeur** : Réduit l'overfitting
6. ✅ **Random state fixe** : Reproductibilité des résultats

## 📊 Visualisations

### Matrice de confusion
```python
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.savefig('results/confusion_matrix.png')
```

### Importance des features
```python
feature_importance.plot(kind='barh', x='feature', y='importance', figsize=(10, 8))
plt.title('Top Features pour la Détection de Fraude')
plt.savefig('results/feature_importance.png')
```

## 🚧 Améliorations Futures

- [ ] Tester d'autres algorithmes (XGBoost, LightGBM, Neural Networks)
- [ ] Optimisation des hyperparamètres (GridSearchCV, RandomizedSearchCV)
- [ ] Feature engineering avancé
- [ ] Validation croisée stratifiée
- [ ] Déploiement API (Flask/FastAPI)
- [ ] Monitoring en production
- [ ] Interface utilisateur web

## 📝 Notes Importantes

### Data Leakage - Colonnes à exclure :
- `trans_num` : Identifiant unique de transaction
- `unix_time` : Timestamp exact
- `trans_date_trans_time` : Date/heure complète
- Toute colonne calculée APRÈS la fraude

### Métriques Prioritaires :
Pour la détection de fraude, privilégier :
1. **Recall** : Ne pas manquer de vraies fraudes (coût élevé)
2. **Precision** : Éviter trop de faux positifs (expérience client)
3. **F1-Score** : Équilibre global

## 👥 Contribution

Les contributions sont les bienvenues ! 

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit les changements (`git commit -m 'Ajout nouvelle feature'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📧 Contact

Pour toute question ou suggestion :
- Email : votre.email@example.com
- GitHub : [@votre-username](https://github.com/votre-username)

## 🙏 Remerciements

- Dataset source : [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/)
- Librairie imbalanced-learn pour SMOTE
- Communauté scikit-learn

---

**⚡ Dernière mise à jour :** Novembre 2025  
**📌 Version :** 1.0.0
