
import pandas as pd


class InputValidationError(Exception):
    """Custom exception for input validation errors."""
    pass


def validate_and_prepare_input(input_df: pd.DataFrame, model):
    """
    Validates input dataframe against model expected features.
    Returns a clean dataframe ready for prediction.
    """

    if not isinstance(input_df, pd.DataFrame):
        raise InputValidationError("Input must be a pandas DataFrame.")

    # Get expected feature names from trained XGBoost model
    try:
        expected_features = model.get_booster().feature_names
    except Exception:
        raise InputValidationError("Unable to retrieve model feature names.")

    # -------------------------
    # 1 Check missing columns
    # -------------------------
    missing_cols = set(expected_features) - set(input_df.columns)
    if missing_cols:
        raise InputValidationError(
            f"Missing required columns: {list(missing_cols)}"
        )

    # -------------------------
    # 2 Check extra columns
    # -------------------------
    extra_cols = set(input_df.columns) - set(expected_features)
    if extra_cols:
        raise InputValidationError(
            f"Unexpected columns provided: {list(extra_cols)}"
        )

    # -------------------------
    # 3 Enforce numeric types
    # -------------------------
    for col in expected_features:
        if not pd.api.types.is_numeric_dtype(input_df[col]):
            raise InputValidationError(
                f"Column '{col}' must be numeric."
            )

    # -------------------------
    # 4 Reorder columns safely
    # -------------------------
    input_df = input_df[expected_features]

    return input_df
