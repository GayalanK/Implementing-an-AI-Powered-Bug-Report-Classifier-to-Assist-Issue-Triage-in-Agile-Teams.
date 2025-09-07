from flask import Flask, request, render_template, send_file
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from werkzeug.utils import secure_filename
import os

# --------------------------
# Flask app initialization
# --------------------------
app = Flask(__name__, template_folder='templates')

# --------------------------
# Load pre-trained model & tokenizer
# --------------------------
MODEL_PATH = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# --------------------------
# File upload settings
# --------------------------
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --------------------------
# Helper functions
# --------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_text(text):
    """Tokenize and encode the text for the model."""
    if not isinstance(text, str) or not text.strip():  # Ensure text is a non-empty string
        raise ValueError("Input text must be a non-empty string.")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    return inputs

def predict_bug_report(summary):
    """Predict Severity, Type, and Confidence for a single bug summary."""
    try:
        inputs = preprocess_text(summary)
    except ValueError as e:
        raise ValueError(f"Invalid input: {str(e)}")

    # Move input tensors to the device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Predicted class (highest logit)
    severity_pred = torch.argmax(logits, dim=-1).item()
    type_pred = severity_pred  # Same output for type for simplicity

    # Compute softmax and confidence percentage
    confidence = torch.nn.functional.softmax(logits, dim=-1)
    confidence_percentage = f"{confidence[0][severity_pred].item() * 100:.3f}%"

    # Map numeric predictions to labels
    severity_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    type_mapping = {0: 'UI Bug', 1: 'Performance Issue', 2: 'Logic Bug'}

    severity = severity_mapping.get(severity_pred, 'Unknown')
    bug_type = type_mapping.get(type_pred, 'Unknown')

    return severity, bug_type, confidence_percentage

# --------------------------
# Routes
# --------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    predictions = None  # Initialize predictions in case there's no input
    error_message = None  # Initialize error message
    if request.method == "POST":
        # --- Option 1: CSV file upload ---
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                input_df = pd.read_csv(file_path)

                # Check if the CSV file is empty
                if input_df.empty:
                    error_message = "The uploaded CSV file is empty. Please upload a valid file."
                    return render_template("index.html", error=error_message)

                if 'Summary' not in input_df.columns:
                    error_message = "CSV file must contain a 'Summary' column."
                    return render_template("index.html", error=error_message)

                try:
                    # Apply prediction on the 'Summary' column and create new columns for severity, type, and confidence
                    input_df['Predicted Severity'], input_df['Predicted Type'], input_df['Prediction Confidence'] = zip(
                        *input_df['Summary'].apply(predict_bug_report)
                    )

                    # Filter the relevant columns: 'Summary', 'Predicted Severity', 'Predicted Type', 'Prediction Confidence'
                    result_df = input_df[['Summary', 'Predicted Severity', 'Predicted Type', 'Prediction Confidence']]

                    # Save the filtered result to CSV
                    results_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'bug_report_results.csv')
                    result_df.to_csv(results_csv_path, index=False)

                    # Return the template with download link
                    return render_template("result.html", input_df=result_df, download_url='/download', predictions=None)

                except Exception as e:
                    return f"Error during prediction: {str(e)}"

        # --- Option 2: Single summary input (user enters text) ---
        if 'summary' in request.form:
            summary = request.form['summary']
            if summary:
                try:
                    severity, bug_type, confidence = predict_bug_report(summary)
                    predictions = {
                        "Summary": summary,
                        "Predicted Severity": severity,
                        "Predicted Type": bug_type,
                        "Prediction Confidence": confidence
                    }

                    # Ensure predictions_df is created only when predictions exist
                    if predictions:
                        # Save the results to CSV
                        results_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'bug_report_results.csv')
                        predictions_df = pd.DataFrame([predictions])
                        predictions_df.to_csv(results_csv_path, index=False)

                        # Return the template with prediction results
                        return render_template("result.html", predictions=predictions, download_url='/download', input_df=None)

                except ValueError as e:
                    error_message = f"Invalid input: {str(e)}"
                    return render_template("index.html", error=error_message)

    # GET request renders the input form page
    return render_template("index.html", error=error_message)

@app.route("/download", methods=["GET"])
def download_results():
    # Path to the saved CSV file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'bug_report_results.csv')

    # Check if the file exists and send it for download
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)

    return "Error: File not found."

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
