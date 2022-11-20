from flask import Flask, jsonify, request

# import Oracle from src.models.inference
from src.models.inference import Oracle


# Create the application instance
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'wav', 'mp3'}


def allowed_file(filename):
    # Check if the file is allowed
    # *.wav or *.mp3
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API Endpoint: /predict


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        # Get the audio file from the request
        audio_file = request.files["file"]

        # Input Checking
        if audio_file is None or audio_file.filename == "":
            return jsonify({"error": "No file was uploaded."})

        if not allowed_file(audio_file.filename):
            return jsonify({"error": "File type not supported."})

        try:
            waveform = audio_file.read()
            TheOracle = n.Oracle()
            predictions = TheOracle.get_predictions(waveform)
            data = {"Confidence Interval": predictions}
            return jsonify(data)

        except:
            return jsonify({"error": "File could not be read."})
