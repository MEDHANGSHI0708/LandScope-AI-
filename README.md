# LandScope-AI-
AI-powered system that leverages Google Earth Engine (GEE) and machine learning to classify land cover with high precision. It analyzes Sentinel-2 and Landsat-9 imagery, using NDVI, NDBI, and MNDWI indices to detect vegetation, urban areas, and water bodies efficiently.










https://github.com/user-attachments/assets/4b1cda34-07f3-4593-8af5-7a66959ad10b







# CLASSIFICATION OVER AMAZON BASIN

![Screenshot from 2025-02-18 16-34-35](https://github.com/user-attachments/assets/34550e5e-a209-4b38-a9ee-2ed9d30bba79)











# CLASSSIFIACTION OVER TOKYO CITY IN JAPAN

![Screenshot from 2025-02-20 21-29-58](https://github.com/user-attachments/assets/6c994ffd-3aa9-4ae7-b7cc-6115b044e02f)












# Landsat Image Classification & Clustering

## Overview
This project applies machine learning techniques to classify satellite imagery data using **Google Earth Engine (GEE)** and a **Random Forest Classifier** with clustering methods. The model processes **NDVI (Normalized Difference Vegetation Index), NDBI (Normalized Difference Built-up Index), and MNDWI (Modified Normalized Difference Water Index)** to segment and classify land cover types.

---

## Mathematical Intuition & Formulas

### 1. **Feature Extraction: NDVI, NDBI, and MNDWI**
Satellite images are processed to extract three critical indices:

- **NDVI (Vegetation Index)**:
  \[ NDVI = \frac{(NIR - RED)}{(NIR + RED)} \]
  Measures vegetation health by comparing Near Infrared (NIR) and Red bands.

- **NDBI (Built-up Area Index)**:
  \[ NDBI = \frac{(SWIR - NIR)}{(SWIR + NIR)} \]
  Helps in identifying built-up areas using Short-Wave Infrared (SWIR) and Near Infrared (NIR) bands.

- **MNDWI (Water Index)**:
  \[ MNDWI = \frac{(GREEN - SWIR)}{(GREEN + SWIR)} \]
  Enhances water body detection by comparing Green and SWIR bands.

---

### 2. **Data Preprocessing**
- **Handling Outliers:** Uses **RobustScaler**, which is robust to outliers by scaling based on interquartile range (IQR).
- **Feature Selection:** Selects only **NDVI, NDBI, and MNDWI**.
- **Missing Values:** Drops missing or infinite values to maintain data integrity.

---

### 3. **Unsupervised Learning: K-Means Clustering**
Before classification, the model clusters data into groups using **K-Means**:
- Assigns each sample to the nearest cluster centroid using **Euclidean Distance**:
  \[ d = \sqrt{\sum (x_i - \mu_i)^2} \]
- Updates centroids iteratively to minimize within-cluster variance:
  \[ J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2 \]
- Number of clusters (**k**) is set to **4**.

---

### 4. **Handling Imbalanced Classes: SMOTE**
- **Synthetic Minority Over-sampling Technique (SMOTE)** generates synthetic data points for underrepresented classes.
- Uses **k-Nearest Neighbors (k-NN)** to create artificial samples in the feature space.

---

### 5. **Supervised Learning: Random Forest Classifier**
Once clusters are assigned, the model applies a **Random Forest Classifier (RFC)** for classification:
- **Bootstrap Aggregation (Bagging):** Combines multiple decision trees trained on different data subsets.
- **Feature Selection:** Each tree randomly selects a subset of features.
- **Splitting Criteria:**
  - **Gini Impurity:**
    \[ G = 1 - \sum_{i=1}^{c} p_i^2 \]
  - **Entropy (Optional):**
    \[ H = - \sum_{i=1}^{c} p_i \log_2 (p_i) \]
- **Final Prediction:** Majority voting among trees determines the final class.

---

### 6. **Hyperparameter Tuning: GridSearchCV**
- **Grid Search** optimizes RFC hyperparameters:
  - `n_estimators`: Number of trees.
  - `max_depth`: Maximum depth of trees.
  - `min_samples_split`: Minimum samples needed to split a node.
  - `min_samples_leaf`: Minimum samples per leaf.
- **Cross-validation (CV = 5)** ensures robust model performance.

---

## Workflow Summary
1. **Data Collection:** Load satellite indices from CSV.
2. **Preprocessing:** Remove NaN, scale data using RobustScaler.
3. **Clustering (K-Means):** Assigns initial labels to data.
4. **SMOTE:** Balances dataset for classification.
5. **Train-Test Split:** 80% training, 20% testing.
6. **Random Forest Training:** Optimized using GridSearchCV.
7. **Evaluation:** Uses classification report & confusion matrix.
8. **Model Saving:** Saves trained model and scaler using `joblib`.

---









