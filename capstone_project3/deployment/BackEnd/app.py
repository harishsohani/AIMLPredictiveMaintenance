
# import for loading model
import joblib

# import pandas
import pandas as pd

# for creating APIs to serve requests 
from flask import Flask, request, jsonify

# import validation to check input
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

        prediction  = model.predict(validated_df)[0]
        probability = model.predict_proba(validated_df)[0, 1]

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "probability": float(probability)
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


# Define an endpoint to predict sales for Super Kart
@pred_mainteanance_api.post ('/v1/EngPredMaintenanceForBatch')
def predict_need_maintenance_for_batch ():
    
    # validate request (json)
    # if input is valid - return prediction
    # in case of error - return appropriate error
    try:
        # Get the uploaded CSV file from the request
        file = request.files.get('file')

        if file is None:
            return jsonify({
                "status": "error",
                "error_type": "input_error",
                "message": "File not provided"
            }), 400
            
        if file.filename == "":
            return jsonify({
                "status": "error",
                "error_type": "input_error",
                "message": "No file selected"
            }), 400        
    
        # Read the file into a DataFrame
        input_df = pd.read_csv (file)

        if input_df.empty:
            return jsonify({
                "status": "error",
                "error_type": "input_error",
                "message": "Uploaded file is empty"
            }), 400

        
        # Process the data to clean up and make it ready for prediction
        # mostly we will use the file with same format as given in problem statement for batch prediction
        
        # remove/drop engine condition column if present
        input_df.drop(columns=['Engine Condition'], inplace=True, errors='ignore')

        # update column names to replace spaces with underscore
        input_df.columns = input_df.columns.str.replace(' ', '_')

        # Convert int → float
        int_columns = input_df.select_dtypes(include=['int64']).columns
        input_df[int_columns] = input_df[int_columns].astype('float64')
        
        # Validate entire batch
        validated_df = validate_and_prepare_input(input_df, model)

        # predict for given input
        predictions   = model.predict(validated_df)
        probabilities = model.predict_proba(validated_df)[:, 1]

        # Convert numpy array → Python list
        prediction_list    = predictions.tolist()
        probabilities_list = probabilities.tolist()

        return jsonify({
                "status": "success",                     # overall batch status
                "total_records": len(prediction_list),
                "predictions": prediction_list,          # simple list version
                "probabilities": probabilities_list,     # simple list version
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
