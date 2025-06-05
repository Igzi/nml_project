import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, filtfilt # Use filtfilt as in the example, or sosfiltfilt if preferred
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split # Optional: for local validation
from sklearn.metrics import classification_report # Optional: for local validation

# Assuming torch and other necessary imports from the original script are available
# We might need to reinstall or adjust imports based on the execution environment
try:
    # Import necessary items from the user's example code context
    from pathlib import Path
    from torch.utils.data import DataLoader
    # The EEGDataset class definition is needed. Assuming it's defined elsewhere
    # If not, we'll need its definition. For now, assume it exists.
    # from seiz_eeg.dataset import EEGDataset # Placeholder from user example
    
    # --- Mock EEGDataset if not available ---
    # This is a simplified version for demonstration if the original class isn't loaded
    # It assumes the basic structure from the user's description
    from torch.utils.data import Dataset
    class EEGDataset(Dataset):
        def __init__(self, clips_df, signals_root, signal_transform=None, prefetch=False, return_id=False, label_col='label'):
            self.clips_df = clips_df
            self.signals_root = Path(signals_root)
            self.signal_transform = signal_transform
            self.return_id = return_id
            self.label_col = label_col

            self.data_cache = {}
            self.signal_files = {} # Cache loaded signal files

            # Determine if it's train or test based on columns
            self.is_test = label_col not in clips_df.columns

            if prefetch:
                print("Prefetching data...")
                for idx in range(len(self.clips_df)):
                     # In a real scenario, pre-load into self.data_cache
                     # For simplicity here, we'll still load on demand but cache signal files
                     pass # Preloading logic would go here if fully implemented

        def __len__(self):
            return len(self.clips_df)

        def __getitem__(self, idx):
            segment_info = self.clips_df.iloc[idx]
            segment_id = self.clips_df.index[idx]

            # Construct the full path to the signal file
            signal_file_path = self.signals_root / segment_info['signals_path']

            # Load signal file if not cached
            if str(signal_file_path) not in self.signal_files:
                 self.signal_files[str(signal_file_path)] = pd.read_parquet(signal_file_path)

            signals_df = self.signal_files[str(signal_file_path)]


            start_time = segment_info['start_time']
            end_time = segment_info['end_time']
            sampling_rate = segment_info['sampling_rate']

            # Calculate start and end indices
            start_index = int(start_time * sampling_rate)
            end_index = int(end_time * sampling_rate)

            # Extract the segment (ensure it's inclusive of start, exclusive of end, matching typical slicing)
            # Adjust indices if necessary based on how time maps to samples
            segment_data = signals_df.iloc[start_index:end_index].values.astype(np.float32) # shape: [time_steps, channels]

            if self.signal_transform:
                 # Pass sampling rate if needed by the transform
                 segment_data = self.signal_transform(segment_data, fs=sampling_rate)


            if self.is_test or self.return_id:
                # For test set or if ID is requested
                 return segment_data, segment_id
            else:
                # For training set
                label = segment_info[self.label_col]
                return segment_data, label
    # --- End Mock EEGDataset ---

except ImportError as e:
    print(f"Import Error: {e}. Make sure necessary libraries and the EEGDataset class are available.")
    # Exit or handle error appropriately if essential components are missing
    exit()


# Define constants
FS = 250  # Default Sampling frequency (Hz), confirm from data if possible
FREQ_BANDS = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'gamma': [30, 60]}

# --- Hjorth Parameter Calculation ---

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
    Callable class to extract Hjorth parameters for all bands from raw EEG signal segment.
    """
    def __init__(self, freq_bands=FREQ_BANDS, fs=FS, filter_order=4):
        self.freq_bands = freq_bands
        self.fs = fs
        self.filter_order = filter_order

    def _filter(self, data: np.ndarray, band: str):
        """ Filters the data with a band pass filter """
        nyq = 0.5 * self.fs
        
        # Handle frequency limits relative to Nyquist
        low = self.freq_bands[band][0]
        high = self.freq_bands[band][1]

        # Ensure frequencies are within valid range [0, nyq]
        low = max(low, 0.01) # Avoid 0 Hz
        high = min(high, nyq - 0.01) # Avoid Nyquist

        # If the band is too narrow or invalid after adjustment, handle appropriately
        if low >= high:
             print(f"Warning: Band {band} [{self.freq_bands[band][0]}, {self.freq_bands[band][1]}] is invalid or too narrow for fs={self.fs}. Returning original data.")
             # Option 1: Return original data (might skew results)
             # return data 
             # Option 2: Return zeros or NaNs (might be safer)
             return np.zeros_like(data)


        low_norm, high_norm = [x / nyq for x in (low, high)]
        
        # Use sosfiltfilt for stability, requires SOS format output from butter
        sos = butter(self.filter_order, [low_norm, high_norm], btype='bandpass', output='sos')
        # Apply filter along the time axis (axis=0)
        return sosfiltfilt(sos, data, axis=0)

    def __call__(self, data: np.ndarray, fs: int = None):
        """
        Calculates Hjorth parameters for all defined bands and channels.
        Args:
            data (np.ndarray): Input raw EEG segment [time_steps, channels].
            fs (int, optional): Sampling frequency. Overrides the default if provided.

        Returns:
            np.ndarray: Flattened feature vector [num_channels * num_bands * 3].
        """
        if fs is not None:
            # Allow overriding FS per call if needed (e.g., dataset provides it)
             if self.fs != fs:
                 # print(f"Updating Feature Extractor FS from {self.fs} to {fs}") # Optional logging
                 self.fs = fs


        all_features = []
        num_channels = data.shape[1]

        for band in self.freq_bands:
            filtered_data = self._filter(data, band)

            # Calculate Hjorth parameters for the filtered data for all channels at once
            activity = hjorth_activity(filtered_data)
            mobility = hjorth_mobility(filtered_data)
            complexity = hjorth_complexity(filtered_data)

            # Append features for this band (shape: [3, num_channels])
            band_features = np.vstack([activity, mobility, complexity])
            all_features.append(band_features)

        # Stack features for all bands (shape: [num_bands, 3, num_channels])
        stacked_features = np.stack(all_features, axis=0)

        # Reshape to [num_bands * 3, num_channels]
        reshaped_features = stacked_features.reshape(-1, num_channels)

        # Transpose and flatten to get [num_channels * num_bands * 3] feature vector
        # Order will be: [ch1_delta_act, ch1_delta_mob, ch1_delta_com, ch1_theta_act, ..., chN_gamma_com]
        # Or, more likely, [ch1_delta_act, ch2_delta_act, ..., chN_delta_act, ch1_delta_mob, ...]
        # Let's flatten channel-wise first, then parameter-wise
        # Final shape: [num_bands, 3, num_channels] -> [num_channels, num_bands, 3] -> flatten
        feature_vector = stacked_features.transpose(2, 0, 1).flatten()


        # Alternatively, flatten preserving band grouping per channel:
        # [ch1_delta_act, ch1_delta_mob, ch1_delta_com, ch1_theta_act, ..., ch1_gamma_com, ch2_delta_act, ...]
        # feature_vector = stacked_features.transpose(2,0,1).flatten() # This might be what we want? Let's test.
        # Let's stick to the structure: [Activity_Delta_Ch1, ..., Activity_Delta_ChN, Mobility_Delta_Ch1, ..., Complexity_Gamma_ChN]
        # This corresponds to reshaping `reshaped_features` column-wise (Fortran order)
        # feature_vector = reshaped_features.flatten(order='F')

        return feature_vector

# --- Data Loading and Preparation ---

# Assume DATA_ROOT and clips_tr/clips_te are loaded as in the user's example
# Example DATA_ROOT (replace with actual path)
# DATA_ROOT = Path("./data/") # Or the path provided: "/home/stnikoli/nml_project/data/"
# Make sure the path exists and contains the correct structure
data_path = "/home/stnikoli/nml_project/data/" # Use the path from the user prompt
DATA_ROOT = Path(data_path)

if not DATA_ROOT.exists():
     print(f"Error: Data root path does not exist: {DATA_ROOT}")
     print("Please ensure the 'data' directory with train/test subfolders is correctly placed.")
     # You might want to stop execution here if data is missing
     exit()


print("Loading segment metadata...")
clips_tr = pd.read_parquet(DATA_ROOT / "train/segments.parquet")
clips_te = pd.read_parquet(DATA_ROOT / "test/segments.parquet")
print(f"Loaded {len(clips_tr)} training segments and {len(clips_te)} test segments.")


# Instantiate the feature extractor
hjorth_extractor = HjorthFeatureExtractor(fs=FS) # Use the default FS=250

# Create datasets with the Hjorth feature extractor
# Important: Ensure the EEGDataset loads RAW signals (no time_filtering or fft_filtering beforehand)
print("Creating training dataset with Hjorth features...")
dataset_tr_hjorth = EEGDataset(
    clips_tr,
    signals_root=DATA_ROOT / "train",
    signal_transform=hjorth_extractor,
    prefetch=True, # Use prefetch if memory allows, might be slow otherwise
    return_id=False # Get labels for training
)

print("Creating test dataset with Hjorth features...")
dataset_te_hjorth = EEGDataset(
    clips_te,
    signals_root=DATA_ROOT / "test",
    signal_transform=hjorth_extractor,
    prefetch=True,
    return_id=True # Get IDs for submission
)

# Load all data into memory for scikit-learn training
# This might take time and memory depending on the dataset size

def load_all_data(dataset):
    features = []
    labels_or_ids = []
    print(f"Loading all data from dataset ({len(dataset)} samples)...")
    # Use DataLoader for potential batching benefits during loading, though batch_size=1 is fine too
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2) # Adjust batch_size/num_workers
    
    # --- Original Iteration Method (Simpler, might be slow) ---
    # for i in tqdm(range(len(dataset)), desc="Loading data"):
    #     data, label_or_id = dataset[i]
    #     features.append(data)
    #     labels_or_ids.append(label_or_id)
    # --- End Original ---

    # --- DataLoader Method ---
    from tqdm import tqdm
    for batch_data, batch_labels_or_ids in tqdm(loader, desc="Loading data"):
         # Assuming data is already processed numpy array by dataset
         # If data comes as tensors, convert: batch_data.numpy()
         features.extend(list(batch_data.numpy())) # Store features as list of numpy arrays
         labels_or_ids.extend(list(batch_labels_or_ids)) # Store labels/ids
    # --- End DataLoader Method ---


    return np.array(features), labels_or_ids # Convert features list to a single large NumPy array


X_train, y_train = load_all_data(dataset_tr_hjorth)
# Test data returns IDs instead of labels
X_test, test_ids = load_all_data(dataset_te_hjorth)

print(f"Training data shape: X={X_train.shape}, y={len(y_train)}")
print(f"Test data shape: X={X_test.shape}, ids={len(test_ids)}")

# Convert y_train list to numpy array for sklearn compatibility
y_train = np.array(y_train)


# --- Model Training ---

print("Setting up and training RandomForestClassifier...")

# Create a pipeline with scaling and the classifier
# StandardScaler is often recommended for SVM and Logistic Regression,
# less critical but can still be beneficial for RandomForest.
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=200,        # Number of trees
        random_state=42,        # For reproducibility
        n_jobs=-1,              # Use all available CPU cores
        max_depth=20,           # Limit tree depth to prevent overfitting
        min_samples_leaf=5,     # Minimum samples per leaf node
        class_weight='balanced' # Adjust for potential class imbalance
        ))
])

# Train the model
model_pipeline.fit(X_train, y_train)

print("Training complete.")

# Optional: Evaluate on a validation set if you split X_train/y_train earlier
# X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
# model_pipeline.fit(X_train_split, y_train_split)
# y_pred_val = model_pipeline.predict(X_val)
# print("\nValidation Set Performance:")
# print(classification_report(y_val, y_pred_val))


# --- Prediction and Submission ---

print("Generating predictions on the test set...")
test_predictions = model_pipeline.predict(X_test)

print("Predictions generated.")

# Format for submission
# Use the same ID correction function as in the user's example if needed
def remove_underlines(s):
    s = s.replace("___", "###")
    s = s.replace("_", "")
    s = s.replace("###", "_")
    return s

# Correct IDs if necessary (assuming test_ids are strings needing correction)
try:
    # Check if the first ID looks like it needs correction
    if isinstance(test_ids[0], str) and ("_" in test_ids[0]):
         print("Applying ID correction...")
         corrected_ids = [remove_underlines(i) for i in test_ids]
    else:
         print("IDs seem okay, not applying correction.")
         corrected_ids = test_ids # Use original IDs if they seem fine
except Exception as e:
    print(f"Warning: Could not process IDs for correction - {e}. Using raw IDs.")
    corrected_ids = test_ids


# Create submission DataFrame
submission_df = pd.DataFrame({'id': corrected_ids, 'label': test_predictions})

# Save submission file
submission_filename = "submission_hjorth_rf.csv"
submission_df.to_csv(submission_filename, index=False)

print(f"Kaggle submission file generated: {submission_filename}")

# --- Optional: Try other models ---
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
#
# print("\nTraining Logistic Regression...")
# lr_pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', C=0.1))])
# lr_pipeline.fit(X_train, y_train)
# print("Logistic Regression Training complete.")
# test_predictions_lr = lr_pipeline.predict(X_test)
# submission_df_lr = pd.DataFrame({'id': corrected_ids, 'label': test_predictions_lr})
# submission_df_lr.to_csv("submission_hjorth_lr.csv", index=False)
# print("Saved submission_hjorth_lr.csv")

# print("\nTraining SVM...")
# svm_pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', SVC(random_state=42, class_weight='balanced', C=1.0, gamma='scale'))]) # SVC can be slow
# svm_pipeline.fit(X_train, y_train)
# print("SVM Training complete.")
# test_predictions_svm = svm_pipeline.predict(X_test)
# submission_df_svm = pd.DataFrame({'id': corrected_ids, 'label': test_predictions_svm})
# submission_df_svm.to_csv("submission_hjorth_svm.csv", index=False)
# print("Saved submission_hjorth_svm.csv")