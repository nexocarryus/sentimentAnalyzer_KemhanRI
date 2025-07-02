from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import pandas as pd
import os
import joblib
import glob
from models.preprocess import preprocess

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


model = joblib.load(os.path.join('models', 'lmodel.pkl'))
vectorizer = joblib.load(os.path.join('models', 'lvectorizer.pkl'))


prediction_status = {'status': 'idle'}

@app.route('/')
def index():
    return render_template('interface.html')

@app.route('/halaman_lain')
def halaman_lain():
    return render_template('interface1.html')

@app.route('/analyzing')
def analyzing():
    return render_template('interface2.html')

@app.route('/uploads', methods=['POST'])
def upload_file():
    global prediction_status
    prediction_status['status'] = 'processing'

    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if not file.filename.lower().endswith('.csv'):
        return jsonify({'message': 'File harus dalam format CSV'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

   
    process_file(filepath)

    return jsonify({'message': 'File uploaded successfully', 'redirect_url': url_for('halaman_lain')})

@app.route('/status', methods=['GET'])
def check_status():
    return jsonify(prediction_status)

def process_file(filepath):
    global prediction_status

    try:
        data = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(filepath, encoding='latin1')
        except UnicodeDecodeError:
            data = pd.read_csv(filepath, encoding='iso-8859-1')

    if 'mentions' not in data.columns:
        prediction_status['status'] = 'error'
        return

    processed_data = data['mentions'].apply(preprocess)
    vectorized_data = vectorizer.transform(processed_data)
    predictions = model.predict(vectorized_data)

    data['sentiment'] = predictions
    output_filepath = os.path.join(UPLOAD_FOLDER, 'predictions_' + os.path.basename(filepath))
    data.to_csv(output_filepath, index=False)

    prediction_status['status'] = 'completed'


def get_latest_prediction_file():
    prediction_files = glob.glob(os.path.join(UPLOAD_FOLDER, 'predictions_*.csv'))
    if not prediction_files:
        return None
    return max(prediction_files, key=os.path.getmtime)

@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    latest_file = get_latest_prediction_file()
    if not latest_file:
        return jsonify({'error': 'File prediksi tidak ditemukan'}), 404

    df = pd.read_csv(latest_file)
    return df.to_json(orient='records')

@app.route('/download_predictions', methods=['GET'])
def download_predictions():
    latest_file = get_latest_prediction_file()
    if not latest_file:
        return jsonify({'error': 'File prediksi tidak ditemukan'}), 404

    return send_file(latest_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)