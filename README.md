# LandScope-AI-
AI-powered system that leverages Google Earth Engine (GEE) and machine learning to classify land cover with high precision. It analyzes Sentinel-2 and Landsat-9 imagery, using NDVI, NDBI, and MNDWI indices to detect vegetation, urban areas, and water bodies efficiently.










https://github.com/user-attachments/assets/4b1cda34-07f3-4593-8af5-7a66959ad10b







# CLASSIFICATION OVER AMAZON BASIN

![Screenshot from 2025-02-18 16-34-35](https://github.com/user-attachments/assets/34550e5e-a209-4b38-a9ee-2ed9d30bba79)











# CLASSSIFIACTION OVER TOKYO CITY IN JAPAN

![Screenshot from 2025-02-20 21-29-58](https://github.com/user-attachments/assets/6c994ffd-3aa9-4ae7-b7cc-6115b044e02f)







# Landsat Image Classification & Clustering



## Mathematical Intuition & Formulas

### 1. **Feature Extraction: NDVI, NDBI, and MNDWI**
Satellite images are processed to extract three critical indices:

- **NDVI (Vegetation Index)**:
  ![Screenshot from 2025-02-20 21-51-44](https://github.com/user-attachments/assets/d039abbf-335f-483d-a3c6-7765c0670dab)





- **NDBI (Built-up Area Index)**:
  ![Screenshot from 2025-02-20 21-52-53](https://github.com/user-attachments/assets/25899492-8319-41a1-a12d-f3d2366e95f9)





- **MNDWI (Water Index)**:
  ![Screenshot from 2025-02-20 21-53-36](https://github.com/user-attachments/assets/57dafd8c-a7ee-40f8-933e-ddefc45b9826)


---

### 2. **Data Preprocessing**
- **Handling Outliers:** Uses **RobustScaler**, which is robust to outliers by scaling based on interquartile range (IQR).
- **Feature Selection:** Selects only **NDVI, NDBI, and MNDWI**.
- **Missing Values:** Drops missing or infinite values to maintain data integrity.

---

### 3. **Unsupervised Learning: K-Means Clustering**
Before classification, the model clusters data into groups using **K-Means**:
- Assigns each sample to the nearest cluster centroid using **Euclidean Distance**:


![Screenshot from 2025-02-20 21-54-35](https://github.com/user-attachments/assets/211b15d8-ef1b-46be-ab85-9091608ae575)



  
- Updates centroids iteratively to minimize within-cluster variance:
- Number of clusters (**k**) is set to **4**.
- IDENTIFIED USING THE ELBOW METHOD
- ![Screenshot from 2025-02-15 22-50-59](https://github.com/user-attachments/assets/48b746bb-76ee-480a-b69a-58b85b08e8c3)


---

### 4. **Handling Imbalanced Classes: SMOTE**
- **Synthetic Minority Over-sampling Technique (SMOTE)** generates synthetic data points for underrepresented classes.



- ![Screenshot from 2025-02-16 00-42-36](https://github.com/user-attachments/assets/75554888-7be2-4569-84ea-c7c6557f8842)



- Uses **k-Nearest Neighbors (k-NN)** to create artificial samples in the feature space.

---

### 5. **Supervised Learning: Random Forest Classifier**
Once clusters are assigned, the model applies a **Random Forest Classifier (RFC)** for classification:
- **Bootstrap Aggregation (Bagging):** Combines multiple decision trees trained on different data subsets.
- **Feature Selection:** Each tree randomly selects a subset of features.
- **Splitting Criteria:**
  - **Gini Impurity:**

![Screenshot from 2025-02-20 21-55-48](https://github.com/user-attachments/assets/be44bbcc-854d-4602-92ad-a61313027ef5)


  - **Entropy (Optional):**
  - 
    ![Screenshot from 2025-02-20 21-58-11](https://github.com/user-attachments/assets/cf418e55-c782-4965-83cc-6831b56c7247)


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



























