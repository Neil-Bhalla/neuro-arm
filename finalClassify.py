import numpy as np
from pylsl import StreamInlet, resolve_stream
from joblib import load
import time
import warnings
import serial
from scipy.signal import welch, find_peaks

def detect_artifact_threshold(eeg_sample, threshold=180):
    return any(abs(signal) > threshold for signal in eeg_sample)

def extract_features(eeg_data, sfreq, nperseg=64):
    psd_features, max_amplitude_features, peak_count_features, band_powers = [], [], [], []
    bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 45)}

    for trial in eeg_data:
        freqs, psd = welch(trial, sfreq, nperseg=nperseg)
        psd_mean = np.mean(psd, axis=1)
        max_amplitude = np.max(np.abs(trial), axis=1)
        peak_counts = [len(find_peaks(channel, height=330)[0]) for channel in trial]

        total_power = np.sum(psd, axis=1)
        band_power_ratios = []
        for band in bands.values():
            band_freqs = (freqs >= band[0]) & (freqs <= band[1])
            band_power = np.sum(psd[:, band_freqs], axis=1)
            band_power_ratio = band_power / total_power
            band_power_ratios.append(band_power_ratio)

        combined_features = np.concatenate([psd_mean, max_amplitude, peak_counts, np.concatenate(band_power_ratios)])
        psd_features.append(combined_features)

    return np.array(psd_features)

def connect_to_arduino(port):
    while True:
        try:
            ser = serial.Serial(port, 9600, timeout=1)
            print("Connected to Arduino.")
            return ser
        except serial.SerialException:
            print("Trying to connect to Arduino...")
            time.sleep(1)

def realtime_labeling(num_channels, sfreq, artifact_duration, clf, scaler, arduino_port):
    print("Starting real-time labeling...")
    ser = connect_to_arduino(arduino_port)
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    num_samples = int(artifact_duration * sfreq)

    action_mapping = {0: "Blink", 1: "Bite", 2: "Double Blink", 3: "Double Bite"}

    while True:
        sample, timestamp = inlet.pull_sample()
        if detect_artifact_threshold(sample[:num_channels - 1]):
            artifact_data = []
            for _ in range(num_samples):
                sample, _ = inlet.pull_sample()
                artifact_data.append(sample[:num_channels - 1])

            features = extract_features(np.array([artifact_data]), sfreq)
            features_scaled = scaler.transform(features)
            prediction = clf.predict(features_scaled)[0]
            action = action_mapping.get(prediction, "Unknown")
            print("Prediction:", action)

            try:
                ser.write(str(prediction).encode())
            except serial.SerialException as e:
                print(f"Error communicating with Arduino: {e}")
                ser.close()
                ser = connect_to_arduino(arduino_port)

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    clf = load('svm_classifier.joblib')
    scaler = load('scaler.joblib')

    artifact_duration = 0.5  # seconds
    num_channels = 5
    sfreq = 256  # Hz
    arduino_port = 'COM4'

    realtime_labeling(num_channels, sfreq, artifact_duration, clf, scaler, arduino_port)

if __name__ == "__main__":
    main()
