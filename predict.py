import ee
import geemap
import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path

class LandsatPredictor:
    def __init__(self, project_id, model_path="rf_model.pkl", scaler_path="scaler.pkl"):
        self.project_id = project_id
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.initialize_gee()
        self.load_model_and_scaler()

    def initialize_gee(self):
        try:
            ee.Initialize(project=self.project_id)
            logging.info(f"GEE initialized successfully with project: {self.project_id}")
        except ee.EEException as e:
            logging.error(f"GEE Initialization failed: {str(e)}")
            raise

    def load_model_and_scaler(self):
        try:
            if not self.model_path.exists() or not self.scaler_path.exists():
                raise FileNotFoundError("Model or scaler file not found")

            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            logging.info("Model and scaler loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model or scaler: {str(e)}")
            raise

    def compute_indices(self, image):
        return image.addBands([
            image.normalizedDifference(["B5", "B4"]).rename("NDVI"),
            image.normalizedDifference(["B6", "B5"]).rename("NDBI"),
            image.normalizedDifference(["B3", "B6"]).rename("MNDWI")
        ])

    def get_landsat_image(self, aoi, start_date="2023-01-01", end_date="2023-12-31"):
        try:
            landsat = (ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA")
                       .filterBounds(aoi)
                       .filterDate(start_date, end_date)
                       .sort("CLOUD_COVER")
                       .first())

            if not landsat:
                raise ValueError("No Landsat images found for the specified criteria")

            logging.info("Landsat image successfully retrieved")
            return self.compute_indices(landsat)
        except Exception as e:
            logging.error(f"Failed to get Landsat image: {str(e)}")
            raise

    def extract_features(self, image, aoi):
        try:
            bands = ["NDVI", "NDBI", "MNDWI"]
            samples = image.select(bands).sample(
                region=aoi,
                scale=30,
                numPixels=1000,
                geometries=True
            ).limit(1000)

            sample_dict = samples.getInfo()
            if "features" not in sample_dict or not sample_dict["features"]:
                raise ValueError("No valid samples found in the specified region")

            feature_list = [feat["properties"] for feat in sample_dict["features"]]
            df = pd.DataFrame(feature_list)

            expected_features = ["NDVI", "NDBI", "MNDWI"]
            for feature in expected_features:
                if feature not in df.columns:
                    df[feature] = 0.0

            logging.info("Extracted feature columns before reordering: %s", df.columns.tolist())
            df = df[expected_features]
            logging.info("Reordered feature columns: %s", expected_features)

            return df, samples
        except Exception as e:
            logging.error(f"Feature extraction failed: {str(e)}")
            raise

    def predict(self, aoi):
        try:
            landsat_image = self.get_landsat_image(aoi)
            features, sample_points = self.extract_features(landsat_image, aoi)

            if features.empty:
                raise ValueError("Extracted features are empty. Check AOI or satellite image availability.")

            logging.info("Scaler expects features in this order: %s", self.scaler.feature_names_in_)
            scaled_features = self.scaler.transform(features)
            predictions = self.model.predict(scaled_features)

            logging.info("Predictions completed successfully")
            return predictions, sample_points
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise

    def visualize_predictions(self, aoi, predictions, sample_points):
        try:
            Map = geemap.Map()
            Map.add_basemap('SATELLITE')
            Map.centerObject(aoi, 14)

            # Convert predictions to an Earth Engine FeatureCollection
            predictions_list = ee.List(predictions.tolist())
            sample_list = sample_points.toList(sample_points.size())

            def set_prediction(f):
                index = sample_list.indexOf(f)
                return ee.Feature(f).set("Prediction", predictions_list.get(index))

            features = sample_points.map(set_prediction)
            prediction_fc = ee.FeatureCollection(features)
            pred_raster = prediction_fc.reduceToImage(["Prediction"], ee.Reducer.first())

            # 🔴🟢🔵 Apply a clear classification colormap
            vis_params = {
                "min": 0,
                "max": 2,
                "palette": ["blue", "green", "red"]  # Water, Vegetation, Urban
            }
            Map.addLayer(pred_raster, vis_params, "Predicted Map")
            sentinel = (ee.ImageCollection("COPERNICUS/S2")
                        .filterBounds(aoi)
                        .filterDate("2023-01-01", "2023-12-31")
                        .sort("CLOUDY_PIXEL_PERCENTAGE")
                        .first())

            rgb_vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
            Map.addLayer(sentinel, rgb_vis, "Sentinel-2 RGB")
            # 🌊 Water bodies (Blue)
            water_mask = sentinel.normalizedDifference(["B3", "B8"]).gt(0.1)
            Map.addLayer(water_mask.selfMask(), {"palette": "blue"}, "Water Bodies")

            # 🌿 Vegetation (Green)
            vegetation_mask = sentinel.normalizedDifference(["B8", "B4"]).gt(0.3)
            Map.addLayer(vegetation_mask.selfMask(), {"palette": "green"}, "Vegetation")

            # 🏠 Urban areas (Red)
            urban_mask = sentinel.normalizedDifference(["B11", "B8"]).gt(0.1)
            Map.addLayer(urban_mask.selfMask(), {"palette": "red"}, "Urban Areas")

            Map.to_html("prediction_result.html")
            logging.info("Map saved as prediction_result.html")
        except Exception as e:
            logging.error(f"Visualization failed: {str(e)}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    predictor = LandsatPredictor(project_id="ee-code2modelxx")

    aoi_json = """
    {
        "type": "Polygon",
        "coordinates": [[
            [-60.3, -3.5], [-60.3, -3.49],
            [-60.29, -3.49], [-60.29, -3.5],
            [-60.3, -3.5]
        ]]
    }
    """

    try:
        aoi = ee.Geometry.Polygon(json.loads(aoi_json)["coordinates"])
        clusters, sample_points = predictor.predict(aoi)
        predictor.visualize_predictions(aoi, clusters, sample_points)
        logging.info("Prediction and visualization completed successfully.")
    except Exception as e:
        logging.error(f"Prediction pipeline failed: {str(e)}")
        raise



