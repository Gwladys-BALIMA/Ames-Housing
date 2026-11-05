# Ames Housing Price Prediction: Linear Regression vs. Random Forest

A machine learning project designed to predict house sale prices in Ames, Iowa. This project explores data cleaning, feature engineering, and a comparison between baseline statistical models and ensemble tree based models using Scikit-Learn pipelines.

## 🚀 Project Overview
Predicting house prices is a classic regression problem. However, with real world housing data , we encounter structural anomalies, missing values, and non linear relationships. This project implements an automated data preprocessing workflow to build, evaluate, and interpret predictive models on both a mathematical log scale and a real world currency scale.

### Key Achievements:
* **Preprocessed 81 original columns** by using custom Scikit-Learn `ColumnTransformer` workflows.
* **Handled missing values**: In the data description many of the missing values meant that the house simply does not have that feature , except from two features (LotFrontage , Electrical) that really have missing values. For the 'non real missing values' , we handled them by replacing structural NaNs with  "None". In the other hand , LotFrontage missing values were filled with the median of all houses grouped by the Neighborhood and Electrical missing value was filled with the mode of the houses based on the age of the houses.
* **Eliminated data skew** by applying a natural log transformation (`np.log1p`) to the target variable (`SalePrice`).


## 📊 Model Performance Table

After evaluating both pipelines on a **20% test dataset**, metrics were transformed back into **real dollar currency** (`np.expm1`) to ensure intuitive, real-world business tracking. 

**Linear Regression outperformed Random Forest across all key metrics:**

| Model | R² Score (Real Scale) | RMSE (Real Dollars) | Avg. Error (MAE) |
| :--- | :--- | :--- | :--- |
| **Linear Regression** 🏆 | **0.9172** | **\$20,844.6126** | **~\$14,629.0375** |
| **Random Forest** | **0.8631** | **\$26,805.2341** | **~\$17,913.0171** |

### Why Linear Regression Won:
1. **Target Normalization:** The log transformation successfully resolved target asymmetry and heteroscedasticity, aligning with Linear Regression's assumptions.
2. **Outlier Mitigation:** Removing properties with extreme footprints (>4,000 sq. ft. 'GrLivArea') eliminated leverage points that heavily penalize linear slopes.
3. **Smooth Generalization:** Linear regression excelled at modeling continuous trends across standard price without risk of the overfitting seen in tree ensembles on restricted numeric splits.


## 🔍 Technical Implementation Details

### The Data Preprocessing Pipeline
To guarantee clean data boundaries and avoid data leakage between training and testing splits, all operations were encapsulated into a unified Scikit-Learn Pipeline:

```python
# Numerical features are scaled to prevent high-magnitude metrics from skewing weights
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Categorical features are safely expanded into binary dimensions 
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
```

---

## 📈 Visual Diagnostic Insights

### 1. Residual Analysis (Prediction Error Tracking)
Evaluating the model via a **Residual Plot** mapped the exact dollar errors ($Y - \hat{Y}$) against predicted valuations. 
* **The Normal Tier:** Between $100k and $200k, errors tightly bounded the zero line, proving incredible accuracy for standard properties.
* **The Luxury Tier:** Past $250k, the chart displays a shape (heteroscedasticity), showing that luxury tier properties introduce higher pricing variance.

### 2. Primary Price Drivers (Feature Weights)
Extracting the mathematical coefficients from the winning regression pipeline reveals the concrete impacts on real estate values. Because the model reads log-dollars, coefficients represent direct **percentage shifts**:

* **Zoning Impact:** Commercial Zoning '`MSZoning_C (all)') is the single strongest negative driver, slashing a property's baseline value by roughly **30%**.
* **Neighborhood Premium:** The Crawford neighborhood ('Neighborhood_Crawfor') commands a massive positive price premium of **~15%**, while Meadow Valley ('Neighborhood_MeadowV') drops predicted value by **~15%**.
* **Property Size:** Above ground living area ('GrLivArea') provides a clean, stable **+8% price boost** for every 1 standard deviation increase in size.


## 🛠️ How to Run the Project

### Prerequisites
Ensure your Python environment contains the following dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

### Execution
1. Clone this repository:
   ```bash
   git clone https://github.com/Gwladys-BALIMA/Ames-Housing.git


