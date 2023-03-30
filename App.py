import streamlit as st
import librosa
import numpy as np
import tensorflow as tf

# Load the MLP model
model = tf.keras.models.load_model('modelForPrediction1.sav')

# Define the labels
LABELS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# Define a function to extract features from an audio file
def extract_features(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=22050, duration=3)
    
    # Extract the features
    mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=30)
    hnr = librosa.feature.spectral_harmonic_ratio(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    
    # Concatenate the features
    features = np.concatenate((mfccs, hnr, zcr))
    
    return features.reshape(1, -1)

# Define the Streamlit app
def app():
    st.set_page_config(page_title='Emotion Detection App', page_icon=':microphone:', layout='wide')
    
    # Define the app layout
    st.title('Emotion Detection App')
    st.write('Upload an mp3 file and we will predict the corresponding emotion.')
    audio_file = st.file_uploader('Upload an mp3 file', type=['mp3'])
    
    if audio_file is not None:
        # Extract the features from the audio file
        features = extract_features(audio_file)
        
        # Make a prediction
        prediction = model.predict(features)
        emotion = LABELS[np.argmax(prediction)]
        
        # Show the prediction
        st.write(f'Predicted Emotion: {emotion}')
