import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from seiz_eeg.dataset import EEGDataset
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm

def evaluate(model, 
              clips, 
              signals_root, 
              num_epochs, 
              learning_rate, 
              batch_size,
              threshold = 0.5,
              prefetch=True,
              signal_transform=None,
              model_args=None,
              criterion=nn.BCEWithLogitsLoss(),
              k_folds=5):
    """
    Perform cross-validation on the EEG dataset and compute the F1 score.

    Parameters:
        model: Callable that returns a new instance of the model.
        clips: DataFrame containing the dataset, including signal paths and labels.
        signals_root: Path to the root directory of the signals.
        signal_transform: Function to preprocess signals (e.g., FFT filtering).
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        batch_size: Batch size for training and validation.
        threshold: Threshold for binary classification (default: 0.5).
        prefetch: Boolean indicating whether to prefetch data during dataset loading.
        criterion: Loss function to use during training (default: BCEWithLogitsLoss).
        k_folds: Number of folds for cross-validation.

    Returns:
        avg_f1_score: Average F1 score across all folds.
        std_f1_score: Standard deviation of F1 scores across all folds.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a complete dataset first and apply transform once
    complete_dataset = EEGDataset(
        clips,
        signals_root=signals_root,
        signal_transform=signal_transform,
        prefetch=prefetch
    )
    
    # Extract unique patients from the dataset
    patients = clips['signals_path'].unique()  # Assuming 'patient_id' column exists

    # Perform K-Fold cross-validation on patients
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(patients)):
        print(f"Fold {fold + 1}/{k_folds}")

        # Split patients into training and validation sets for this fold
        train_patients = patients[train_idx]
        val_patients = patients[val_idx]

        # Filter clips based on the split patients
        train_clips = clips[clips['signals_path'].isin(train_patients)]
        val_clips = clips[clips['signals_path'].isin(val_patients)]
        
        # Get indices for training and validation from the complete dataset
        train_indices = [i for i, path in enumerate(clips['signals_path']) if path in train_patients]
        val_indices = [i for i, path in enumerate(clips['signals_path']) if path in val_patients]
        
        # Create subset datasets using the precomputed transforms
        train_dataset = Subset(complete_dataset, train_indices)
        val_dataset = Subset(complete_dataset, val_indices)

        # Create DataLoaders
        train_loader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=batch_size
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            shuffle=False,
            batch_size=batch_size
        )

        # Initialize model, loss function, and optimizer
        model_fold = model(**model_args)  # Create a new instance of the model for each fold
        model_fold.to(device)
        optimizer = optim.Adam(model_fold.parameters(), lr=learning_rate)

        # Training loop
        for epoch in tqdm(range(num_epochs), desc=f"Training Fold {fold + 1}"):
            model_fold.train()
            for signals, labels in train_loader:
                signals, labels = signals.float().to(device), labels.float().unsqueeze(1).to(device)
                optimizer.zero_grad()
                outputs = model_fold(signals)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Validation phase
        model_fold.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.float().to(device), labels.float().to(device)
                val_logits = model_fold(signals)
                val_probs = torch.sigmoid(val_logits)  # Apply sigmoid for binary classification
                val_preds = (val_probs > threshold).int()  # Convert probabilities to binary predictions

                # Collect all labels and predictions
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(val_preds.cpu().numpy())

        # Compute F1 score for this fold
        fold_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Fold {fold + 1} F1 Score: {fold_f1:.4f}")
        fold_results.append(fold_f1)

        del model_fold, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()

    # Calculate average F1 score across all folds
    avg_f1_score = np.mean(fold_results)
    std_f1_score = np.std(fold_results)

    print(f"Cross-Validation Average F1 Score: {avg_f1_score:.4f}")
    print(f"Cross-Validation F1 Score Standard Deviation: {std_f1_score:.4f}")

    return avg_f1_score, std_f1_score

def evaluate_sklearn(model, 
                        clips, 
                        signals_root, 
                        threshold=0.5,
                        prefetch=True,
                        signal_transform=None,
                        model_args=None,
                        k_folds=5):
    """
    Evaluate a sklearn-compatible model using patient-based cross-validation and compute the F1 score.
    
    Parameters:
        model: Sklearn-compatible model class.
        clips: DataFrame containing the dataset, including signal paths and labels.
        signals_root: Path to the root directory of the signals.
        threshold: Threshold for binary classification (default: 0.5).
        prefetch: Boolean indicating whether to prefetch data during dataset loading.
        signal_transform: Function to preprocess signals.
        model_args: Dictionary of arguments to pass to the model constructor.
        k_folds: Number of folds for cross-validation.
        
    Returns:
        avg_f1_score: Average F1 score across all folds.
        std_f1_score: Standard deviation of F1 scores across all folds.
    """
    # Create a complete dataset first and apply transform once
    complete_dataset = EEGDataset(
        clips,
        signals_root=signals_root,
        signal_transform=signal_transform,
        prefetch=prefetch
    )
    
    # Extract unique patients from the dataset
    patients = clips['signals_path'].unique()

    # Perform K-Fold cross-validation on patients
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(patients)):
        print(f"Fold {fold + 1}/{k_folds}")

        # Split patients into training and validation sets for this fold
        train_patients = patients[train_idx]
        val_patients = patients[val_idx]

        # Get indices for training and validation from the complete dataset
        train_indices = [i for i, path in enumerate(clips['signals_path']) if path in train_patients]
        val_indices = [i for i, path in enumerate(clips['signals_path']) if path in val_patients]
        
        # Extract features and labels for training
        X_train = []
        y_train = []
        for idx in train_indices:
            signal, label = complete_dataset[idx]
            X_train.append(signal.flatten())  # Flatten for sklearn models
            y_train.append(label.item())
        
        # Extract features and labels for validation
        X_val = []
        y_val = []
        for idx in val_indices:
            signal, label = complete_dataset[idx]
            X_val.append(signal.flatten())  # Flatten for sklearn models
            y_val.append(label.item())
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        # Initialize and train model
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        
        # Compute F1 score for this fold
        fold_f1 = f1_score(y_val, y_pred, average='macro')
        print(f"Fold {fold + 1} F1 Score: {fold_f1:.4f}")
        fold_results.append(fold_f1)

    # Calculate average F1 score across all folds
    avg_f1_score = np.mean(fold_results)
    std_f1_score = np.std(fold_results)

    print(f"Cross-Validation Average F1 Score: {avg_f1_score:.4f}")
    print(f"Cross-Validation F1 Score Standard Deviation: {std_f1_score:.4f}")

    return avg_f1_score, std_f1_score