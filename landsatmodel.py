import ee
import geemap
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import logging



# print statements
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
class LandsatModel:
    def __init__(self, project_id):
        self.project_id = project_id
        self.initialize_gee()
        self.model = None
        self.scaler = None
    def initialize_gee(self):
        """GEE InItilaized"""
        try:
            ee.Initialize(project=self.project_id)
            logging.info(f"GEE initialized successfully with project: {self.project_id}")
        except ee.EEException as e:
            logging.error(f"GEE Initialization failed: {str(e)}")
            raise

    def load_and_preprocess_data(self, csv_path):
        """Load, clean, and preprocess the dataset with validation"""
        try:
            df = pd.read_csv(csv_path)

            # Validate required columns
            required_columns = ["NDVI", "NDBI", "MNDWI"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Expected: {required_columns}")

            # Remove rows with NaN or Infinite values
            df = df.dropna(subset=required_columns)
            df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]

            # Select features
            X = df[required_columns]

            # Handle outliers using RobustScaler
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)

            return X_scaled, df

        except Exception as e:
            logging.error(f"Data preprocessing failed: {str(e)}")
            raise

    def train_model(self, X_scaled, df, n_clusters=4):
        """Train the model with cross-validation and hyperparameter tuning"""
        try:
            from sklearn.cluster import KMeans

            # Apply K-means clustering to generate labels
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df["Cluster"] = kmeans.fit_predict(X_scaled)
            y = df["Cluster"]

            # Handle imbalanced classes using SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
            )

            # Define parameter grid for RandomForest
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            }

            # Create and train model with GridSearchCV
            rf = RandomForestClassifier(random_state=42)
            self.model = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            logging.info("\nBest parameters: %s", self.model.best_params_)
            logging.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
            logging.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

            return self.model, self.scaler

        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise

    def save_model(self, model_path="rf_model.pkl", scaler_path="scaler.pkl"):
        """Save the trained model and scaler"""
        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            logging.info(f"Model and scaler saved successfully to {model_path} and {scaler_path}")
        except Exception as e:
            logging.error(f"Failed to save model: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize and train model
    model_trainer = LandsatModel(project_id="ee-code2modelxx")

    try:
        # Load and preprocess data
        X_scaled, df = model_trainer.load_and_preprocess_data("Satellite imagery2.csv")

        # Train model
        model, scaler = model_trainer.train_model(X_scaled, df)

        # Save model and scaler
        model_trainer.save_model()
    except Exception as e:
        logging.error(f"Training pipeline failed: {str(e)}")
        raise
expected_features = ["NDVI", "NDBI", "MNDWI"]

# ✅ Ensure correct order and fill missing ones with 0.0
features = features.reindex(columns=expected_features, fill_value=0.0)



