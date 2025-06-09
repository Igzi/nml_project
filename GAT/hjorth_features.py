import numpy as np
from scipy.signal import butter, sosfiltfilt

# Define constants
FS = 250  # Sampling frequency (Hz)
FREQ_BANDS = {
    'delta': [0.5, 4], 
    'theta': [4, 8], 
    'alpha': [8, 13], 
    'beta': [13, 30], 
    'gamma': [30, 60]
}

def hjorth_activity(data: np.ndarray):
    """Calculates Hjorth activity (variance)."""
    return np.var(data, axis=0)

def hjorth_mobility(data: np.ndarray):
    """Calculates Hjorth mobility."""
    diff1 = np.diff(data, axis=0)
    var_diff1 = np.var(diff1, axis=0)
    var_data = np.var(data, axis=0)
    # Avoid division by zero
    mobility = np.sqrt(var_diff1 / (var_data + 1e-8))
    return mobility

def hjorth_complexity(data: np.ndarray):
    """Calculates Hjorth complexity."""
    diff1 = np.diff(data, axis=0)
    diff2 = np.diff(diff1, axis=0)
    var_diff1 = np.var(diff1, axis=0)
    var_diff2 = np.var(diff2, axis=0)
    # Avoid division by zero
    mobility = hjorth_mobility(data)
    complexity = np.sqrt(var_diff2 / (var_diff1 + 1e-8)) / (mobility + 1e-8)
    return complexity

class HjorthFeatureExtractor:
    """
    Extract Hjorth parameters for all frequency bands from raw EEG signal.
    Returns features suitable for GAT input: [electrodes, features]
    """
    def __init__(self, freq_bands=FREQ_BANDS, fs=FS, filter_order=4):
        self.freq_bands = freq_bands
        self.fs = fs
        self.filter_order = filter_order

    def _filter(self, data: np.ndarray, band: str):
        """Filters the data with a band pass filter"""
        nyq = 0.5 * self.fs
        
        # Handle frequency limits relative to Nyquist
        low = self.freq_bands[band][0]
        high = self.freq_bands[band][1]

        # Ensure frequencies are within valid range [0, nyq]
        low = max(low, 0.01)  # Avoid 0 Hz
        high = min(high, nyq - 0.01)  # Avoid Nyquist

        # If the band is too narrow or invalid after adjustment
        if low >= high:
            print(f"Warning: Band {band} is invalid for fs={self.fs}. Skipping.")
            return np.zeros_like(data)

        low_norm, high_norm = [x / nyq for x in (low, high)]
        
        # Use sosfiltfilt for stability
        sos = butter(self.filter_order, [low_norm, high_norm], btype='bandpass', output='sos')
        return sosfiltfilt(sos, data, axis=0)

    def __call__(self, data: np.ndarray, fs: int = None):
        """
        Calculate Hjorth parameters for all frequency bands.
        
        Args:
            data: Input EEG segment [time_steps, channels]
            fs: Sampling frequency (optional override)
            
        Returns:
            np.ndarray: Features [channels, num_bands * 3]
                       Shape: [19, 15] for 5 bands × 3 parameters
        """
        if fs is not None and self.fs != fs:
            self.fs = fs

        all_features = []
        num_channels = data.shape[1]

        for band in self.freq_bands:
            filtered_data = self._filter(data, band)

            # Calculate Hjorth parameters for all channels
            activity = hjorth_activity(filtered_data)      # [channels]
            mobility = hjorth_mobility(filtered_data)      # [channels]
            complexity = hjorth_complexity(filtered_data)  # [channels]

            # Stack parameters for this band: [3, channels]
            band_features = np.vstack([activity, mobility, complexity])
            all_features.append(band_features)

        # Stack all bands: [num_bands, 3, channels]
        stacked_features = np.stack(all_features, axis=0)
        
        # Reshape to [channels, num_bands * 3]
        # Order: [ch1_delta_act, ch1_delta_mob, ch1_delta_com, ch1_theta_act, ...]
        features_per_channel = stacked_features.transpose(2, 0, 1).reshape(num_channels, -1)
        
        return features_per_channel

def create_hjorth_transforms():
    """Create Hjorth-based signal transformation functions."""
    hjorth_extractor = HjorthFeatureExtractor()
    
    def hjorth_transform(x: np.ndarray) -> np.ndarray:
        """Transform EEG signal to Hjorth features for GAT."""
        return hjorth_extractor(x).T
    
    return hjorth_transform, hjorth_transform  # Same for train and test
