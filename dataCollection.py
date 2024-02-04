import numpy as np
from pylsl import StreamInlet, resolve_stream
import time
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

        psd_features.append(psd_mean)
        max_amplitude_features.append(max_amplitude)
        peak_count_features.append(peak_counts)
        band_powers.append(np.concatenate(band_power_ratios))

    combined_features = np.concatenate([psd_features, max_amplitude_features, peak_count_features, band_powers], axis=1)
    return combined_features

def collect_data_for_action(action, num_trials, artifact_duration, num_channels, sfreq):
    print(f"Collecting data for {action}.")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    all_features = []
    num_samples = int(artifact_duration * sfreq)

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials} for {action}. Get ready and press Enter to start.")
        artifact_detected = False

        while not artifact_detected:
            sample, timestamp = inlet.pull_sample()
            if detect_artifact_threshold(sample[:num_channels - 1]):
                print(f"Recording {action}. 0.3 seconds recording starting now...")
                artifact_data = []
                for _ in range(num_samples):
                    sample, _ = inlet.pull_sample()
                    artifact_data.append(sample[:num_channels - 1])

                features = extract_features(np.array([artifact_data]), sfreq)
                all_features.append(features.flatten())
                artifact_detected = True
                print("Recording completed.")
                time.sleep(1)

    return np.array(all_features)

def main():
    actions = ["blink", "bite", "double_blink", "double_bite"]
    artifact_duration = 0.5
    num_trials = 10
    num_channels = 5
    sfreq = 256

    for action in actions:
        action_features = collect_data_for_action(action, num_trials, artifact_duration, num_channels, sfreq)
        np.save(f'{action}_features.npy', action_features)
        print(f"\nFeature collection for {action} completed.")

if __name__ == "__main__":
    main()
