import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance


os.makedirs('output', exist_ok=True)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess data
    """
    print(f"Loading data: {file_path}")
    # Read CSV file
    df = pd.read_csv(file_path, skipinitialspace=True)
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Remove unnecessary columns (IP addresses, ports, and protocol)
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
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def plot_learning_curve(estimator, X, y, model_name, cv=5):
    """
    Plot learning curve for a model
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.15)
    plt.plot(train_sizes, test_mean, label='Cross-validation score', color='green', marker='s')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='green', alpha=0.15)
    
    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'output/learning_curve_{model_name}.png')
    plt.close()

def plot_roc_curve(y_test, y_pred_prob, model_name):
    """
    Plot ROC curve with proper class labels
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

def train_knn(X_train, y_train, X_test, y_test, feature_names):
    """
    Train KNN model
    """
    print("\nTraining KNN model...")
    start_time = time.time()
    
    # Use grid search to find the best K value
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15]}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_knn = grid_search.best_estimator_
    best_k = grid_search.best_params_['n_neighbors']
    print(f"Best K value: {best_k}")
    
    # Plot learning curve
    plot_learning_curve(best_knn, X_train, y_train, "KNN")
    
    # Evaluate the model on the test set
    y_pred = best_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get probability predictions if possible
    try:
        y_pred_prob = best_knn.predict_proba(X_test)
        
        # Plot ROC and PR curves
        plot_roc_curve(y_test, y_pred_prob, "KNN")
        plot_precision_recall_curve(y_test, y_pred_prob, "KNN")
    except:
        print("Could not generate ROC or PR curves for KNN (predict_proba not available)")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"KNN model training time: {training_time:.2f} seconds")
    print(f"KNN model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('KNN Model Confusion Matrix')
    plt.savefig('output/confusion_matrix_KNN.png')
    plt.close()
    
    # Visualize KNN feature importance using permutation importance
    perm_importance = permutation_importance(best_knn, X_test, y_test, n_repeats=10, random_state=42)
    perm_importances = perm_importance.importances_mean
    plot_feature_importance(perm_importances, feature_names, "KNN")
    
    return best_knn, accuracy, training_time

def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    """
    Train Random Forest model
    """
    print("\nTraining Random Forest model...")
    start_time = time.time()
    
    # Use grid search to find the best parameters
    param_grid = {
        'n_estimators': [2, 5, 10],
        'max_depth': [None, 2, 5, 10],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    
    # Plot learning curve
    plot_learning_curve(best_rf, X_train, y_train, "RandomForest")
    
    # Evaluate the model on the test set
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get probability predictions
    y_pred_prob = best_rf.predict_proba(X_test)
    
    # For multi-class, we need to provide label encoder for proper class names
    label_encoder = LabelEncoder()
    y_test_enc = label_encoder.fit_transform(y_test)
    
    # Plot ROC and PR curves
    plot_roc_curve(y_test, y_pred_prob, "RandomForest")
    plot_precision_recall_curve(y_test, y_pred_prob, "RandomForest")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Random Forest model training time: {training_time:.2f} seconds")
    print(f"Random Forest model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Random Forest Model Confusion Matrix')
    plt.savefig('output/confusion_matrix_RandomForest.png')
    plt.close()
    
    # Feature importance analysis
    plot_feature_importance(best_rf.feature_importances_, feature_names, "RandomForest")
    
    return best_rf, accuracy, training_time

def feature_importance(importances, feature_names, model_name, top_n=10):
    """
    Analyze and visualize feature importance
    """
    # Get feature importance
    importances = importances
    indices = np.argsort(importances)[::-1]
    
    # Select top features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    
    # Convert feature_names to list for integer indexing
    feature_names_list = list(feature_names)
    top_names = [feature_names_list[i] for i in top_indices]
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n), top_importances, align='center')
    plt.xticks(range(top_n), top_names, rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title(f'{model_name} Model - Top {top_n} Important Features')
    plt.tight_layout()
    plt.savefig(f'output/feature_importance_{model_name}.png')
    plt.close()
    
    print(f"\nTop {top_n} Important Features for {model_name}:")
    for i in range(top_n):
        print(f"{i+1}. {top_names[i]}: {top_importances[i]:.4f}")

def train_1d_cnn(X_train, y_train, X_test, y_test):
    """
    Train 1D CNN model based on the paper architecture
    """
    print("\nTraining 1D CNN model...")
    start_time = time.time()
    
    # Reshape input data for CNN [samples, timesteps, features]
    # According to the paper, reshape to 1D vector
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Convert labels to categorical
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    num_classes = len(np.unique(y_train))
    y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=num_classes)
    
    # Build the CNN model according to the paper
    model = Sequential()
    
    # Conv Layer 1: 32 filters, ReLU, MaxPooling, LRN
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    # Note: Keras doesn't have built-in LRN, we'll use BatchNormalization as an alternative
    model.add(tf.keras.layers.BatchNormalization())
    
    # Conv Layer 2: 64 filters, ReLU, MaxPooling, LRN
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.BatchNormalization())
    
    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Adding dropout for regularization
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Print model summary
    model.summary()
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train_reshaped, 
        y_train_categorical,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_reshaped, y_test_categorical, verbose=0)
    
    # Get predictions
    y_pred_prob = model.predict(X_test_reshaped)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred)
    
    end_time = time.time()
    training_time = end_time - start_time
        
    print(f"1D CNN model training time: {training_time:.2f} seconds")
    print(f"1D CNN model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('1D CNN Model Confusion Matrix')
    plt.savefig('output/confusion_matrix_CNN.png')
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
    plt.savefig('output/training_history_CNN.png')
    plt.close()
    
    # Visualize CNN layer filters
    try:
        # Get the first convolutional layer filters
        filters, biases = model.layers[0].get_weights()
        
        # Normalize filter values to 0-1 so we can visualize them
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        
        # Plot filters
        n_filters = min(32, filters.shape[2])  # Show up to 32 filters
        plt.figure(figsize=(12, 8))
        
        for i in range(n_filters):
            plt.subplot(4, 8, i+1)
            plt.imshow(filters[:, :, i].reshape(3, 1), cmap='viridis', aspect='auto')
            plt.axis('off')
            
        plt.suptitle('CNN First Layer Filters')
        plt.tight_layout()
        plt.savefig('output/cnn_filters_CNN.png')
        plt.close()
    except:
        print("Could not visualize CNN filters")
    
    # For visualization of CNN feature importance, we can use Grad-CAM or similar techniques
    # This is more complex and may require additional libraries
    
    # Try to create a saliency map for a sample (simplified)
    try:
        import tensorflow.keras.backend as K
        
        # Get a sample from test set
        sample_idx = 0
        sample = X_test_reshaped[sample_idx:sample_idx+1]
        
        # Define a function to compute the gradient of the output w.r.t the input
        output_idx = np.argmax(y_test_categorical[sample_idx])
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.output[:, output_idx], model.get_layer('conv1d').output]
        )
        
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(sample)
            loss = predictions[0]
            
        # Compute gradients
        grads = tape.gradient(loss, conv_output)
        
        # Compute feature importance
        pooled_grads = K.mean(grads, axis=(0, 1))
        
        # Create a heatmap
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        
        # Normalize the heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        
        plt.figure(figsize=(10, 3))
        plt.imshow(heatmap[0].reshape(1, -1), aspect='auto', cmap='viridis')
        plt.title('CNN Activation Heatmap')
        plt.xlabel('Feature Position')
        plt.colorbar(label='Activation')
        plt.tight_layout()
        plt.savefig('output/cnn_heatmap_CNN.png')
        plt.close()
    except:
        print("Could not create CNN activation heatmap")
    
    # Plot ROC curve
    try:
        plot_roc_curve(y_test, y_pred_prob, "CNN")
        plot_precision_recall_curve(y_test, y_pred_prob, "CNN")
    except:
        print("Could not generate ROC or PR curves for CNN")
    
    return model, accuracy, training_time

def plot_feature_importance(importances, feature_names, model_name, top_n=10):
    """
    统一的特征重要性可视化函数
    
    参数:
        importances: 特征重要性值数组
        feature_names: 特征名称列表
        model_name: 模型名称
        top_n: 显示前N个重要特征
    """
    # 获取索引排序
    indices = np.argsort(importances)[::-1]
    
    # 选择前N个特征
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    
    # 确保特征名称是列表格式
    feature_names_list = list(feature_names)
    top_names = [feature_names_list[i] for i in top_indices]
    
    # 绘制特征重要性条形图
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n), top_importances, align='center')
    plt.xticks(range(top_n), top_names, rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title(f'{model_name} Model - Top {top_n} Important Features')
    plt.tight_layout()
    plt.savefig(f'output/feature_importance_{model_name}.png')
    plt.close()
    
    # 打印重要特征信息
    print(f"\nTop {top_n} Important Features for {model_name}:")
    for i in range(top_n):
        print(f"{i+1}. {top_names[i]}: {top_importances[i]:.4f}")

def main():
    # Load and preprocess data
    file_path = './output/basic_features.csv'
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(file_path)
    
    # Train KNN model
    print("\n" + "="*50)
    print("Training KNN Model")
    print("="*50)
    knn_results = train_knn(X_train, y_train, X_test, y_test, feature_names)
    
    # Train Random Forest model
    print("\n" + "="*50)
    print("Training Random Forest Model")
    print("="*50)
    rf_results = train_random_forest(X_train, y_train, X_test, y_test, feature_names)
    
    # Train 1D CNN model
    print("\n" + "="*50)
    print("Training 1D CNN Model")
    print("="*50)
    cnn_results = train_1d_cnn(X_train, y_train, X_test, y_test)
    
    print("\nModel training and evaluation completed!")
    print(f"Results and visualizations saved to the 'output' directory")

if __name__ == "__main__":
    main()
