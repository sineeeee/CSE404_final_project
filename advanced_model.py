import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, Input, Concatenate, Bidirectional, GRU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


os.makedirs('output', exist_ok=True)

def load_and_preprocess_advanced_data(file_path):
    """
    Load and preprocess advanced features data
    
    Parameters:
        file_path (str): Path to the advanced features CSV file
        
    Returns:
        tuple: (X_train_reshaped, X_test_reshaped, y_train_categorical, y_test_categorical, y_train, y_test, label_encoder, num_classes)
    """
    print(f"Loading advanced features data: {file_path}")
    
    # Read CSV file
    df = pd.read_csv(file_path, skipinitialspace=True)
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Remove unnecessary columns (IP addresses and ports)
    columns_to_drop = ['Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol']
    df = df.drop(columns=columns_to_drop)
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"Number of missing values: {missing_values}")
    
    # Handle missing values if any
    if missing_values > 0:
        df = df.dropna()
        print(f"Dataset shape after removing missing values: {df.shape}")
    
    # Check and handle infinite and extreme values
    print("Checking for infinite and extreme values...")
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Check statistics for each column
    print("Data statistics:")
    print(df.describe().T)
    
    # Handle outliers for each column using the IQR method
    for column in df.select_dtypes(include=[np.number]).columns:
        if column != 'label':  # Don't process the label column
            # Calculate Q1 and Q3
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Set lower and upper bounds
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Clip values outside the range
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    # Check for missing values again (possibly from inf conversion)
    missing_after = df.isnull().sum()
    if missing_after.sum() > 0:
        print(f"Number of missing values after handling infinite values: {missing_after.sum()}")
        print("Missing value distribution:")
        print(missing_after[missing_after > 0])
        
        # Fill missing values only for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if missing_after[col] > 0:
                # Fill missing values with median
                df[col] = df[col].fillna(df[col].median())
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Get number of classes
    num_classes = len(np.unique(y_train))
    
    # Convert labels to categorical
    y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=num_classes)
    
    # Reshape data for temporal models
    # We have 10 windows, each with 8 features
    n_features = 8  # Number of features per window
    n_windows = 10  # Number of windows
    
    # Reshape training and test sets
    X_train_reshaped = reshape_for_temporal_model(X_train_scaled, n_windows, n_features)
    X_test_reshaped = reshape_for_temporal_model(X_test_scaled, n_windows, n_features)
    
    print(f"Reshaped training set shape: {X_train_reshaped.shape}")
    print(f"Reshaped test set shape: {X_test_reshaped.shape}")
    
    return X_train_reshaped, X_test_reshaped, y_train_categorical, y_test_categorical, y_train, y_test, label_encoder, num_classes

def reshape_for_temporal_model(X, n_windows=10, n_features=8):
    """
    Reshape data for temporal models [samples, time_steps, features]
    
    Parameters:
        X (numpy.ndarray): Input data
        n_windows (int): Number of windows
        n_features (int): Number of features per window
        
    Returns:
        numpy.ndarray: Reshaped data
    """
    samples = X.shape[0]
    X_reshaped = np.zeros((samples, n_windows, n_features))
    
    for i in range(samples):
        for j in range(n_windows):
            # Starting index for current window features
            start_idx = j * n_features
            # Extract features for current window
            X_reshaped[i, j, :] = X[i, start_idx:start_idx + n_features]
    
    return X_reshaped

def build_cnn_lstm_model(input_shape, num_classes):
    """
    Build CNN-LSTM hybrid model
    
    Parameters:
        input_shape (tuple): Input data shape (time_steps, features)
        num_classes (int): Number of classes
        
    Returns:
        tensorflow.keras.Model: Compiled model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # CNN part
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # LSTM part
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_cnn_gru_model(input_shape, num_classes):
    """
    Build CNN-GRU hybrid model
    
    Parameters:
        input_shape (tuple): Input data shape (time_steps, features)
        num_classes (int): Number of classes
        
    Returns:
        tensorflow.keras.Model: Compiled model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # CNN part
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # GRU part
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = GRU(64)(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_roc_curve(y_test, y_pred_prob, model_name):
    """
    Plot ROC curve with proper class labels
    
    Parameters:
        y_test: True labels
        y_pred_prob: Predicted probabilities
        model_name: Model name for the plot title and filename
    """
    # Check if we're dealing with multi-class classification
    if isinstance(y_pred_prob, np.ndarray) and y_pred_prob.ndim == 2 and y_pred_prob.shape[1] > 1:
        # Multi-class classification
        n_classes = y_pred_prob.shape[1]
        
        # One-hot encode the labels for multi-class ROC
        label_encoder = LabelEncoder()
        y_test_enc = label_encoder.fit_transform(y_test)
        y_test_onehot = to_categorical(y_test_enc, n_classes)
        
        # Get the class names
        class_names = np.unique(y_test)
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_onehot[:, i], y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f'output/roc_curve_{model_name}.png')
        plt.close()
    else:
        # Binary classification
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f'output/roc_curve_{model_name}.png')
        plt.close()

def plot_precision_recall_curve(y_test, y_pred_prob, model_name):
    """
    Plot precision-recall curve with proper class labels
    
    Parameters:
        y_test: True labels
        y_pred_prob: Predicted probabilities
        model_name: Model name for the plot title and filename
    """
    # Check if we're dealing with multi-class classification
    if isinstance(y_pred_prob, np.ndarray) and y_pred_prob.ndim == 2 and y_pred_prob.shape[1] > 1:
        # Multi-class classification
        n_classes = y_pred_prob.shape[1]
        
        # One-hot encode the labels for multi-class precision-recall
        label_encoder = LabelEncoder()
        y_test_enc = label_encoder.fit_transform(y_test)
        y_test_onehot = to_categorical(y_test_enc, n_classes)
        
        # Get the class names
        class_names = np.unique(y_test)
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_test_onehot[:, i], y_pred_prob[:, i])
            avg_precision = average_precision_score(y_test_onehot[:, i], y_pred_prob[:, i])
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {avg_precision:.2f})')
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.savefig(f'output/precision_recall_curve_{model_name}.png')
        plt.close()
    else:
        # Binary classification
        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        avg_precision = average_precision_score(y_test, y_pred_prob)
        
        plt.plot(recall, precision, lw=2, label=f'AP = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.savefig(f'output/precision_recall_curve_{model_name}.png')
        plt.close()

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, raw_y_test, model_name, epochs=50, batch_size=32):
    """
    Train and evaluate model
    
    Parameters:
        model: Model to train
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
        raw_y_test: Original test labels (non-encoded)
        model_name: Model name for plots and output
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        tuple: (training history, accuracy, training time)
    """
    print(f"\nTraining {model_name} model...")
    start_time = time.time()
    
    # Print model summary
    model.summary()
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001),
        ModelCheckpoint(f'output/{model_name}_best_model.h5', save_best_only=True, monitor='val_accuracy')
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"{model_name} model training time: {training_time:.2f} seconds")
    print(f"{model_name} model accuracy: {accuracy:.4f}")
    
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Create label encoder to convert back to original labels
    label_encoder = LabelEncoder()
    label_encoder.fit(raw_y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(raw_y_test, y_pred_labels))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(raw_y_test, y_pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(raw_y_test), yticklabels=np.unique(raw_y_test))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{model_name} Model Confusion Matrix')
    plt.savefig(f'output/confusion_matrix_{model_name}.png')
    plt.close()
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig(f'output/training_history_{model_name}.png')
    plt.close()
    
    # Try to visualize model attention (simplified)
    try:
        # Get first convolutional layer weights
        conv_weights = model.layers[1].get_weights()[0]  # First Conv1D layer
        
        # Normalize weights for visualization
        weights_min, weights_max = conv_weights.min(), conv_weights.max()
        normalized_weights = (conv_weights - weights_min) / (weights_max - weights_min)
        
        # Plot channel-wise weights
        plt.figure(figsize=(12, 6))
        plt.imshow(np.mean(normalized_weights, axis=2).T, cmap='viridis', aspect='auto')
        plt.colorbar(label='Normalized Weight')
        plt.title(f'{model_name} Convolutional Filter Weights')
        plt.xlabel('Filter')
        plt.ylabel('Time Step')
        plt.savefig(f'output/filter_weights_{model_name}.png')
        plt.close()
    except:
        print(f"Could not visualize convolutional filters for {model_name}")
    
    # Try to create a heatmap of activations
    try:
        import tensorflow.keras.backend as K
        
        # Create a model to get intermediate layer outputs
        layer_outputs = [layer.output for layer in model.layers if isinstance(layer, Conv1D) or isinstance(layer, LSTM) or isinstance(layer, GRU)]
        activation_model = Model(inputs=model.input, outputs=layer_outputs)
        
        # Get a sample input
        sample_idx = 0
        sample = X_test[sample_idx:sample_idx+1]
        
        # Get activations
        activations = activation_model.predict(sample)
        
        # Plot first layer activations (assuming it's a Conv1D layer)
        first_layer_activation = activations[0]
        plt.figure(figsize=(12, 6))
        plt.imshow(np.mean(first_layer_activation[0], axis=1).reshape(1, -1), cmap='viridis', aspect='auto')
        plt.colorbar(label='Activation')
        plt.title(f'{model_name} First Layer Activation')
        plt.xlabel('Time Step')
        plt.savefig(f'output/activation_heatmap_{model_name}.png')
        plt.close()
    except:
        print(f"Could not create activation heatmap for {model_name}")
    
    # Plot ROC and PR curves
    try:
        plot_roc_curve(raw_y_test, y_pred_prob, model_name)
        plot_precision_recall_curve(raw_y_test, y_pred_prob, model_name)
    except Exception as e:
        print(f"Could not generate ROC or PR curves for {model_name}: {e}")
    
    return history, accuracy, training_time

def main():
    """Main function"""
    # Load and preprocess advanced features data
    file_path = './output/advanced_features.csv'
    X_train, X_test, y_train_cat, y_test_cat, y_train, y_test, label_encoder, num_classes = load_and_preprocess_advanced_data(file_path)
    
    # Get input shape
    input_shape = X_train.shape[1:]
    
    # Build and train CNN-LSTM model
    print("\n" + "="*50)
    print("Training CNN-LSTM Model")
    print("="*50)
    cnn_lstm_model = build_cnn_lstm_model(input_shape, num_classes)
    cnn_lstm_results = train_and_evaluate_model(
        cnn_lstm_model, X_train, y_train_cat, X_test, y_test_cat, y_test, 'CNN_LSTM'
    )
    
    # Build and train CNN-GRU model
    print("\n" + "="*50)
    print("Training CNN-GRU Model")
    print("="*50)
    cnn_gru_model = build_cnn_gru_model(input_shape, num_classes)
    cnn_gru_results = train_and_evaluate_model(
        cnn_gru_model, X_train, y_train_cat, X_test, y_test_cat, y_test, 'CNN_GRU'
    )
    
    print("\nAdvanced model training and evaluation completed!")
    print(f"Results and visualizations saved to the 'output' directory")

if __name__ == "__main__":
    main()
