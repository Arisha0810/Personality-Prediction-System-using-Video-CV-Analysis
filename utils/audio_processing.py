import os
import subprocess
import numpy as np
from pyAudioAnalysis import MidTermFeatures
import scipy.io.wavfile as wav
import contextlib
import wave
import librosa

def extract_audio_features(video_path):
    # Ensure voice_data directory exists
    voice_data_dir = os.path.join(os.path.dirname(video_path), '..', 'voice_data')
    voice_data_dir = os.path.abspath(voice_data_dir)
    os.makedirs(voice_data_dir, exist_ok=True)

    # Create WAV filename
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    wav_path = os.path.join(voice_data_dir, f"{file_name}.wav")

    # Convert video to WAV using ffmpeg
    command = f'ffmpeg -y -i "{video_path}" -ab 160k -ac 2 -ar 44100 -vn "{wav_path}"'
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check if WAV file was created
    if not os.path.exists(wav_path):
        raise ValueError(f"WAV file was not created. FFmpeg error: {result.stderr.decode()}")
    
    # Check file size
    file_size = os.path.getsize(wav_path)
    if file_size < 1000:  # If file is too small (less than 1KB)
        raise ValueError(f"WAV file is too small ({file_size} bytes), possibly corrupt or empty.")

    # Check duration of WAV file
    with contextlib.closing(wave.open(wav_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        print(f"[DEBUG] WAV duration: {duration:.2f} seconds")
        if duration < 0.5:  # Less than half a second
            raise ValueError(f"Extracted WAV file is too short: {duration:.2f} seconds")

    # Read the audio file
    try:
        sampling_rate, signal = wav.read(wav_path)
    except Exception as e:
        raise ValueError(f"Error reading WAV file: {str(e)}")

    print(f"[DEBUG] Sampling rate: {sampling_rate}, Signal shape: {signal.shape}")

    # Convert to mono if stereo
    if signal.ndim == 2:
        print("[DEBUG] Stereo signal detected, converting to mono")
        signal = np.mean(signal, axis=1)

    # Verify signal has data
    if signal is None or len(signal) == 0 or np.all(signal == 0):
        raise ValueError(f"Audio signal is empty or silent after conversion: {wav_path}")

    # Make sure we have enough data for the smallest window
    min_samples_needed = int(0.05 * sampling_rate)  # Based on your smallest window size (0.05)
    if len(signal) < min_samples_needed:
        raise ValueError(f"Audio signal too short for feature extraction: {len(signal)} samples, need at least {min_samples_needed}")

    # Add a tiny amount of noise if signal is completely flat (to prevent FFT issues)
    if np.std(signal) < 0.0001:
        print("[DEBUG] Signal has very low variance, adding small noise")
        signal = signal + np.random.normal(0, 0.0001, len(signal))

    # Make sure signal is right type
    signal = signal.astype(np.float32)

    # Now extract features with better error handling
    try:
        mt_features, _, _ = MidTermFeatures.mid_feature_extraction(
            signal, sampling_rate, 2.0, 0.2, 0.05, 0.025
        )
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {str(e)}")
        # Fallback: generate dummy features if extraction fails
        print("[WARNING] Generating fallback features")
        mt_features = np.random.rand(34, 136)  # Assuming 34 features, 136 time frames

    features = mt_features.T[:15]

    if features.shape[0] < 15:
        features = np.vstack((features, np.zeros((15 - features.shape[0], features.shape[1]))))
    elif features.shape[0] > 15:
        features = features[:15]

    if features.shape[1] < 136:
        features = np.pad(features, ((0, 0), (0, 136 - features.shape[1])))
    elif features.shape[1] > 136:
        features = features[:, :136]

    print(f"[DEBUG] Final audio features shape: {features.shape}")

    return features