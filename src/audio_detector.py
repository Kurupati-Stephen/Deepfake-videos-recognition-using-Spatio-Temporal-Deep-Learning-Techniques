import librosa
import numpy as np


def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    if len(y) == 0:
        raise ValueError("Empty audio file")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(y)

    features = {
        "mfcc_mean": float(np.mean(mfcc)),
        "mfcc_std": float(np.std(mfcc)),
        "centroid_mean": float(np.mean(spectral_centroid)),
        "zcr_mean": float(np.mean(zero_crossing)),
        "duration": float(len(y) / sr),
    }

    return features


def predict_audio(audio_path):
    features = extract_audio_features(audio_path)

    # simple temporary logic
    fake_score = 0.0

    if features["mfcc_std"] < 40:
        fake_score += 0.2
    if features["zcr_mean"] < 0.03:
        fake_score += 0.2
    if features["centroid_mean"] > 3000:
        fake_score += 0.2
    if features["duration"] < 2:
        fake_score += 0.1

    fake_score = min(fake_score, 0.95)

    if fake_score > 0.5:
        prediction = "Fake"
        confidence = fake_score * 100
        status_message = "Possible synthetic / cloned audio detected"
    else:
        prediction = "Real"
        confidence = (1 - fake_score) * 100
        status_message = "Audio appears authentic"

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "fake_probability": round(fake_score, 4),
        "status_message": status_message
    }