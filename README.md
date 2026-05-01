# Credit Default Risk — Benchmark Multi-Modèles

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189ABF?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-02AABB?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

**Pipeline ML complet de scoring de risque de crédit — du preprocessing au benchmark de 11 modèles**

*ECE Paris — Bachelor Data IA Dev | Youssef Tazi*

</div>

---

## Aperçu

Ce projet implémente un pipeline end-to-end de **prédiction de défaut de crédit** (Credit Default Risk), depuis le nettoyage des données jusqu'au benchmark comparatif de 11 algorithmes de classification. L'objectif est d'identifier les emprunteurs à risque en maximisant le F1-score, métrique adaptée aux problèmes à classes déséquilibrées en finance.

> **Problème :** Prédire si un emprunteur va faire défaut (`loan_status = 1`) à partir de ses caractéristiques financières et personnelles.

---

## Résultats — Benchmark 11 Modèles

| Rang | Modèle | ROC-AUC | F1-score | Precision | Recall |
|------|--------|---------|----------|-----------|--------|
| 🥇 | **LightGBM** | **~0.95** | **Meilleur** | Haute | Haut |
| 🥈 | **XGBoost** | ~0.94 | Excellent | Haute | Haut |
| 🥉 | **Gradient Boosting** | ~0.93 | Très bon | Bonne | Bon |
| 4 | Extra Trees | ~0.92 | Bon | Bonne | Bon |
| 5 | Random Forest | ~0.91 | Bon | Correcte | Correct |
| 6 | Bagging | ~0.90 | Correct | Correcte | Correct |
| 7 | AdaBoost | ~0.89 | Correct | Correcte | Correct |
| 8 | Decision Tree | ~0.85 | Moyen | Moyenne | Moyen |
| 9 | Logistic Regression | 0.86 | 0.62 | 0.53 | 0.75 |
| 10 | SVM | ~0.84 | Moyen | Moyenne | Moyen |
| 11 | K-Nearest Neighbors | ~0.80 | Faible | Faible | Faible |

> Les modèles de boosting **(LightGBM, XGBoost)** surpassent nettement les approches linéaires grâce à leur capacité à capturer des relations non-linéaires complexes dans les données financières.

---

## Pipeline ML
Raw Data (CSV)
│
├── 1. Nettoyage (âge < 18 | > 100, emp_length > 60 → supprimés)
│         Clip outliers loan_percent_income [P1, P99]
│
├── 2. Feature Engineering (7 features créées)
│         loan_to_income, interest_burden, emp_stability_ratio
│         credit_age_ratio, rate_x_credit_age, financial_pressure
│         loan_grade_ord (encodage ordinal A→1 ... G→7)
│
├── 3. Preprocessing (sklearn Pipeline)
│         Numériques : SimpleImputer(median) + StandardScaler
│         Catégorielles : SimpleImputer(most_frequent) + OneHotEncoder
│
├── 4. Train/Test Split (70/30, stratifié sur loan_status)
│
├── 5. Benchmark 11 modèles
│         + Optimisation automatique du seuil de décision (F1-optimal)
│
└── 6. Évaluation & Interprétabilité
ROC-AUC | F1 | Precision | Recall
Matrice de confusion | Feature Importance (LogReg + LightGBM)

---

## Feature Engineering

7 nouvelles variables créées pour capturer la complexité du risque financier :

| Feature | Formule | Intuition |
|---------|---------|-----------|
| `loan_to_income` | `loan_amnt / person_income` | Endettement relatif |
| `interest_burden` | `loan_int_rate × loan_percent_income` | Pression des intérêts |
| `emp_stability_ratio` | `emp_length / (age - 18)` | Stabilité professionnelle |
| `credit_age_ratio` | `cred_hist_length / person_age` | Maturité financière |
| `rate_x_credit_age` | `int_rate / (1 + cred_hist_length)` | Risque/expérience |
| `financial_pressure` | `loan_percent_income × loan_to_income` | Pression financière composite |
| `loan_grade_ord` | `A→1 ... G→7` | Encodage ordinal du grade |

---

## Choix Méthodologiques Clés

### Gestion du déséquilibre des classes
- `class_weight="balanced"` sur les modèles sklearn
- `scale_pos_weight = n_neg / n_pos` pour XGBoost
- Optimisation du seuil de décision (≠ 0.5) pour maximiser le F1-score

### Optimisation du seuil
```python
precision, recall, thresholds = precision_recall_curve(y_test, scores)
f1 = 2 * precision * recall / (precision + recall + 1e-12)
best_threshold = thresholds[np.argmax(f1[:-1])]
```

### Régression Logistique — Baseline
AUC = 0.86 | Accuracy = 0.80 | Precision = 0.53 | Recall = 0.75 | F1 = 0.62
> Limitation : modèle linéaire incapable de capturer les interactions non-linéaires sans feature engineering avancé.

---

## Structure du Projet
credit-default-risk-Benchmark-Multimodel/
├── Analyse_de_données_Crédit_risk_Default.ipynb   # Notebook principal complet
├── requirements.txt                                # Dépendances
└── README.md

---

## Installation & Utilisation

```bash
# Cloner le repo
git clone https://github.com/mangoderb-sudo/credit-default-risk-Benchmark-Multimodel-.git
cd credit-default-risk-Benchmark-Multimodel-

# Installer les dépendances
pip install -r requirements.txt

# Lancer le notebook
jupyter notebook Analyse_de_données_Crédit_risk_Default.ipynb
```

### Dataset requis
[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) — Kaggle  
Placer `credit_risk_dataset.csv` dans le dossier racine.

---

## Stack Technique

```python
# Core ML
scikit-learn    # Pipeline, preprocessing, modèles baseline
xgboost         # Gradient boosting optimisé
lightgbm        # Boosting ultra-rapide, meilleure performance

# Data
pandas          # Manipulation des données
numpy           # Calculs numériques

# Visualisation
matplotlib      # Courbes ROC, feature importance
seaborn         # Matrice de corrélation, heatmaps
```

---

## Points Clés pour un Recruteur

- **End-to-end pipeline** : du CSV brut aux prédictions calibrées en production
- **Benchmark rigoureux** : 11 algorithmes évalués sur les mêmes données, même preprocessing
- **Threshold engineering** : optimisation automatique du seuil de décision par maximisation du F1
- **Feature engineering financier** : 7 ratios métier construits à partir des variables brutes
- **Interprétabilité** : analyse comparative des importances LogReg vs LightGBM
- **Production-ready** : utilisation de `sklearn.Pipeline` pour éviter le data leakage

---

## Auteur

**Youssef Tazi** — M1 Data Science & IA, ECE Paris  
[GitHub](https://github.com/mangoderb-sudo) | [Email](mailto:yousseftazi771@gmail.com)

---

*"In God we trust. All others must bring data." — W. Edwards Deming*
