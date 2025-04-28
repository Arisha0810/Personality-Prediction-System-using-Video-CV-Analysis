from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import numpy as np
from utils.audio_processing import extract_audio_features
from utils.frame_processing import extract_video_frames
from tensorflow.keras.models import load_model
from utils.gemini_summary import generate_personality_summary
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = load_model('model/mod_aud_vid.h5', compile=False)

OCEAN_LABELS = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return 'No video uploaded'
    
    video_file = request.files['video']
    if video_file.filename == '':
        return 'No selected file'
    
    filename = secure_filename(video_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(filepath)

    # Step 1: Extract features
    audio_features = extract_audio_features(filepath)  # (15, 136)
    video_frames = extract_video_frames(filepath)      # (15, 224, 224, 3)

    # Step 2: Expand dims for prediction
    X_audio = np.expand_dims(audio_features, axis=0)
    X_video = np.expand_dims(video_frames, axis=0)

    # Step 3: Predict
    prediction = model.predict([X_video, X_audio])[0]
    result = dict(zip(OCEAN_LABELS, prediction.round(3)))

    # Step 4: Generate summary from Gemini API
    summary = generate_personality_summary(result)

    return render_template('result.html', result=result, summary=summary, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
