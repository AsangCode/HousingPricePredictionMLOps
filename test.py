# test_prediction_pipeline.py
from prediction_pipeline import CustomData, PredictPipeline
import numpy as np

def test_prediction_pipeline():
    try:
        # Sample data for testing
        data = CustomData(
            area=100,
            bedrooms=3,
            bathrooms=2,
            stories=2,
            mainroad='yes',
            guestroom='no',
            basement='yes',
            hotwaterheating='yes',
            airconditioning='no',
            parking=2,
            prefarea='yes',
            furnishingstatus='furnished'
        )

        # Convert the data to DataFrame
        final_data = data.get_data_as_dataframe()
        print("Data as DataFrame:")
        print(final_data)

        # Instantiate the PredictPipeline
        predict_pipeline = PredictPipeline()

        # Perform prediction
        prediction = predict_pipeline.predict(final_data)
        print("Prediction Result:")
        print(prediction)
        assert prediction < 0

        # Assuming model returns an array of predictions
        if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0:
            print(f"Predicted value: {round(prediction[0], 2)}")
        else:
            print("Prediction result is not in expected format.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_prediction_pipeline()
