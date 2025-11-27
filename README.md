# Credit Risk Scorecard: Default Prediction & Strategy Simulation

A complete end-to-end **credit card default prediction** project built with an industry-style **PD model**, **WOE/IV binning**, **logistic scorecard**, and **approval strategy simulation**.

This project follows a real-world credit risk modeling workflow used in banks and consumer finance institutions.

---

## 🚀 Project Highlights

- End-to-end production-style project structure  
- Data preprocessing & feature engineering  
- WOE binning + IV variable selection  
- Logistic regression PD model (AUC ≈ 0.61)  
- Scorecard construction (WOE → Score)  
- Monotonic smoothing for the Score–PD curve  
- Strategy simulation (lenient / baseline / strict)  
- Professional visualizations (ROC, KS, Lift/Gain, Score Distribution, Strategy Tradeoff)

---

## 📁 Project Structure

credit_risk/
│ main.py
│ requirements.txt
│ README.md
│
├── data/
│ ├── raw/ # Raw data (excluded from GitHub)
│ ├── interim/ # Temporary files
│ └── processed/ # Model outputs & results
│
├── figures/ # All generated visualizations
│
├── src/
│ ├── data_prep/ # Data cleaning & preparation
│ ├── features/ # Feature engineering
│ ├── woe_binning/ # Binning, WOE, IV
│ ├── modeling/ # Logistic model, ML challengers, VIF, stepwise
│ ├── scorecard/ # Scorecard building & score transformation
│ ├── validation/ # ROC, KS, lift, monotonicity
│ └── business/ # Business strategy simulation

---

📊 Key Visualizations

### Score Distribution
<img src="figures/score_distribution_pro.png" width="700"/>

---

### Score vs PD (Monotonic Smoothed)
<img src="figures/score_vs_pd_pro.png" width="700"/>

---

### Approval–Risk Tradeoff
<img src="figures/strategy_tradeoff_pro.png" width="700"/>

---

### ROC Curve
<img src="figures/roc_curve.png" width="600"/>

---

### KS Curve
<img src="figures/ks_curve.png" width="600"/>

---

## ⚙️ How to Run

### 1. Install dependencies


### 2. Execute full modeling pipeline


The pipeline includes:

1. Data preparation  
2. Feature engineering  
3. WOE binning + IV selection  
4. Logistic regression PD model  
5. Scorecard generation  
6. Visualization  
7. Strategy simulation  

Outputs will be saved to:

- `data/processed/`
- `figures/`

---

## 📈 Modeling Summary

- Logistic baseline PD model  
- AUC ≈ **0.61**  
- KS ≈ **0.18**  
- ML challengers: RandomForest, GradientBoosting  
- Scorecard built using:
  - WOE-transformed variables  
  - Base score + PDO scaling  
  - Monotonic PD smoothing  

---

## 🧮 Strategy Simulation Results

Three business strategies were evaluated:

| Strategy | Approval Rate | Bad Rate | Expected Loss |
|----------|----------------|----------|----------------|
| Lenient  | ~0.80          | ~0.24    | ~0.32         |
| Baseline | ~0.50          | ~0.26    | ~0.39         |
| Strict   | ~0.20          | ~0.23    | ~0.44         |

### Interpretation

- Strict policy reduces approval rate significantly but does not reduce bad rate as much.  
- Lenient policy increases approvals but raises expected loss.  
- Illustrates real-world credit policy tradeoffs.

---

## 🛠 Tech Stack

- Python  
- pandas / numpy  
- scikit-learn  
- seaborn / matplotlib  
- WOE/IV modeling  
- Logistic regression  
- Gradient Boosting / Random Forest  
- Score scaling and monotonic regression  

---

## 📝 Potential Future Enhancements

- Time-based validation / PSI  
- Reject inference  
- Optimal binning (ChiMerge / MDLP)  
- LightGBM challenger model  
- Score stability monitoring  
- Deployment-ready API template  

---

## 👤 Author

Hongbo Niu  
MSF, Johns Hopkins University – Carey Business School  
Washington D.C.

---

⭐ *If this project was helpful, please give it a star!*

