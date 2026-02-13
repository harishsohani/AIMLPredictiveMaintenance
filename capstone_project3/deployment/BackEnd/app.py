
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from utils.validation import validate_and_prepare_input, InputValidationError

# Initialize Flask app with a name
pred_mainteanance_api = Flask ("Engine Maintenance Predictor")

# Load the trained churn prediction model
model = joblib.load ("best_eng_fail_pred_model.joblib")

# Define a route for the home page
@pred_mainteanance_api.get ('/')
def home ():
    return "Welcome to the Engine Maintenance Prediction!"

# Define an endpoint to predict sales for Super Kart
@pred_mainteanance_api.post ('/v1/EngPredMaintenance')
def predict_need_maintenance ():
    # Get JSON data from the request
    engine_sensor_inputs = request.get_json ()

    # validate request (json)
    # if input is valid - return prediction
    # in case of error - return appropriate error
    try:
        input_json = request.get_json()
        input_df = pd.DataFrame([input_json])

        validated_df = validate_and_prepare_input(input_df, model)

        prediction = model.predict(validated_df)[0]

        return jsonify({
            "status": "success",
            "prediction": int(prediction)
        })

    except InputValidationError as e:
        return jsonify({
            "status": "error",
            "error_type": "validation_error",
            "message": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "status": "error",
            "error_type": "internal_error",
            "message": "Unexpected server error"
        }), 500


# Run the Flask app
if __name__ == "__main__":
    import os
    port = int (os.environ.get("PORT", 7860))
    pred_mainteanance_api.run(host="0.0.0.0", port=port)
