from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the XGBoost model
xgb_model = joblib.load("xgboost_model.pkl")

# Define the exact training columns (order matters)
training_columns = [
    "id",
    "Duration in months",
    "Credit history",
    "Purpose of the credit",
    "Credit amount",
    "Status of savings account/bonds",
    "Present employment(years)",
    "Installment rate in percentage of disposable income",
    "personal_status",
    "Other debtors / guarantors",
    "Present residence since X years",
    "Property",
    "Age in years",
    "Other installment plans (banks/stores)",
    "Housing",
    "Number of existing credits at this bank",
    "Job",
    "Number of people being liable to provide maintenance for",
    "Telephone",
    "Foreign worker",
]

# Mapping from model output to risk categories.
risk_mapping = {
    0: "High Risk",  # e.g., not creditworthy
    1: "Low Risk",  # e.g., creditworthy
}


@app.route("/")
def home():
    return "Welcome to the CreditGuard API!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Convert data to DataFrame
        df = pd.DataFrame([data])

        # Reindex DataFrame to ensure it has all required features in the correct order.
        df = df.reindex(columns=training_columns, fill_value=0)

        # Make prediction using the XGBoost model
        xgb_prediction = xgb_model.predict(df)[0]

        # Map the raw prediction to a human-readable risk category
        risk_category = risk_mapping.get(int(xgb_prediction), "Unknown")

        # Generate a loan recommendation based on the risk category
        if risk_category == "Low Risk":
            loan_recommendation = (
                "Approved: Customer has a low risk profile. Proceed with loan."
            )
        elif risk_category == "High Risk":
            loan_recommendation = (
                "Declined: Customer has a high risk profile. Loan is not recommended."
            )
        else:
            loan_recommendation = "Review: Further analysis required."

        # Build the response including a status code and recommendation
        response = {
            "status_code": 200,
            "Raw_Prediction": int(xgb_prediction),
            "Risk_Category": risk_category,
            "Loan_Recommendation": loan_recommendation,
        }
        return jsonify(response), 200

    except Exception as e:
        error_response = {"status_code": 400, "error": str(e)}
        return jsonify(error_response), 400


if __name__ == "__main__":
    app.run(debug=True)
