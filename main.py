"""
Feature Selection using Variance Threshold and Recursive Feature Elimination (RFE)
for the MEU Mobile KSD Dataset.

This script:
1. Loads and preprocesses the MEU Mobile KSD dataset
2. Implements variance-based feature selection
3. Implements Recursive Feature Elimination (RFE) as an additional feature selection technique
4. Evaluates and compares model performance with different feature selection methods
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the MEU Mobile KSD dataset
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        tuple: X (features), y (target variable)
    """
    print("Loading dataset from:", file_path)
    
    # Load the dataset - skip first few rows which might contain metadata
    # You may need to adjust skiprows parameter based on the actual file structure
    df = pd.read_excel(file_path, skiprows=2)
    
    # Print initial dataset information
    print("\nInitial dataset shape:", df.shape)
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    # Remove rows with NaN values
    df_clean = df.dropna()
    print("\nDataset shape after removing NaN values:", df_clean.shape)
    
    # Based on the dataset structure, the first column (typically labeled as 1) 
    # appears to be the subject/user identifier
    # Use the first column as the target variable
    print("\nUsing the first column as the target variable (Subject/User)")
    y = df_clean.iloc[:, 0]
    X = df_clean.iloc[:, 1:]
    
    # Print the first few values of y to confirm
    print("\nFirst few values of the target variable:")
    print(y.head())
    
    print("\nFeature set shape:", X.shape)
    print("Target variable distribution:")
    print(y.value_counts())
    
    return X, y

def evaluate_model(model, X_test, y_test, title="Model Performance"):
    """
    Evaluate the model and print the results
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target variable
        title: Title for the evaluation results
    
    Returns:
        float: Accuracy score
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{title}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy

def variance_based_feature_selection(X_train, X_test, threshold=0.0):
    """
    Apply variance-based feature selection
    
    Args:
        X_train: Training features
        X_test: Test features
        threshold: Variance threshold
        
    Returns:
        tuple: Transformed X_train and X_test
    """
    # Create a variance threshold selector
    selector = VarianceThreshold(threshold=threshold)
    
    # Fit and transform the training data
    X_train_var = selector.fit_transform(X_train)
    
    # Transform the test data
    X_test_var = selector.transform(X_test)
    
    # Get the feature indices that were selected
    selected_features = selector.get_support(indices=True)
    
    print(f"\nVariance-based Feature Selection (threshold={threshold})")
    print(f"Number of features before selection: {X_train.shape[1]}")
    print(f"Number of features after selection: {X_train_var.shape[1]}")
    
    # Get the selected feature names
    selected_feature_names = [X_train.columns[i] for i in selected_features]
    
    if len(selected_features) < X_train.shape[1]:
        print(f"Features removed: {X_train.shape[1] - len(selected_features)}")
        
        # Print the feature names that were selected
        print("\nSelected features (sample of first 10):")
        for i, feature in enumerate(selected_feature_names[:10]):
            print(f"  {i+1}. {feature}")
        
        if len(selected_feature_names) > 10:
            print(f"  ... and {len(selected_feature_names) - 10} more features")
    else:
        print("No features were removed based on the variance threshold.")
    
    # Convert to DataFrame with selected feature names
    X_train_var_df = pd.DataFrame(X_train_var, columns=selected_feature_names)
    X_test_var_df = pd.DataFrame(X_test_var, columns=selected_feature_names)
    
    return X_train_var_df, X_test_var_df

def rfe_feature_selection(X_train, X_test, y_train, n_features_to_select=None):
    """
    Apply Recursive Feature Elimination (RFE) for feature selection
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target variable
        n_features_to_select: Number of features to select
        
    Returns:
        tuple: Transformed X_train and X_test
    """
    # If n_features_to_select is not specified, select half of the features
    if n_features_to_select is None:
        n_features_to_select = X_train.shape[1] // 2
    
    # Create an SVM classifier for RFE
    svm = SVC(kernel="linear", random_state=42)
    
    # Create the RFE selector
    selector = RFE(estimator=svm, n_features_to_select=n_features_to_select, step=1)
    
    # Fit the RFE selector to the training data
    X_train_rfe = selector.fit_transform(X_train, y_train)
    
    # Transform the test data
    X_test_rfe = selector.transform(X_test)
    
    # Get the feature indices that were selected
    selected_features = np.where(selector.support_)[0]
    
    print(f"\nRecursive Feature Elimination (n_features_to_select={n_features_to_select})")
    print(f"Number of features before selection: {X_train.shape[1]}")
    print(f"Number of features after selection: {X_train_rfe.shape[1]}")
    
    # Print the feature names that were selected
    selected_feature_names = [X_train.columns[i] for i in selected_features]
    print("\nSelected features (sample of first 10):")
    for i, feature in enumerate(selected_feature_names[:10]):
        print(f"  {i+1}. {feature}")
    
    if len(selected_feature_names) > 10:
        print(f"  ... and {len(selected_feature_names) - 10} more features")
    
    # Print feature ranking (only top 10)
    feature_ranks = {str(col): rank for col, rank in zip(X_train.columns, selector.ranking_)}
    sorted_features = sorted(feature_ranks.items(), key=lambda x: x[1])
    print("\nTop 10 features by ranking (lower is better):")
    for i, (feature, rank) in enumerate(sorted_features[:10]):
        print(f"  {i+1}. {feature}: {rank}")
    
    # Convert to DataFrame with selected feature names
    X_train_rfe_df = pd.DataFrame(X_train_rfe, columns=selected_feature_names)
    X_test_rfe_df = pd.DataFrame(X_test_rfe, columns=selected_feature_names)
    
    return X_train_rfe_df, X_test_rfe_df

def main():
    """Main function to run the feature selection process"""
    # File path to the dataset
    file_path = "MEUMobile KSD 2016.xlsx"
    
    # Load and preprocess the data
    X, y = load_and_preprocess_data(file_path)
    
    # Convert all column names to strings to avoid issues with scikit-learn
    X.columns = X.columns.astype(str)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("\nData split: Training set size:", X_train.shape[0])
    print("Data split: Test set size:", X_test.shape[0])
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Since the original column names might be numeric, we'll create generic feature names
    feature_names = [f'Feature_{i+1}' for i in range(X_train.shape[1])]
    
    # Convert back to DataFrame with the generated feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Train the baseline SVM model
    print("\nTraining baseline SVM model...")
    baseline_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    baseline_model.fit(X_train_scaled, y_train)
    
    # Evaluate the baseline model
    baseline_accuracy = evaluate_model(baseline_model, X_test_scaled, y_test, "Baseline SVM Model Performance")
    
    # Calculate and print variance of each feature before selection
    feature_variances = X_train_scaled.var()
    print("\nFeature Variances (top 10 highest):")
    print(feature_variances.sort_values(ascending=False).head(10))
    print("\nFeature Variances (bottom 10 lowest):")
    print(feature_variances.sort_values().head(10))
    
    # Apply variance-based feature selection
    # For this dataset, we'll use a slightly higher threshold to be more selective
    var_threshold = 0.05
    X_train_var, X_test_var = variance_based_feature_selection(X_train_scaled, X_test_scaled, threshold=var_threshold)
    
    # Train the SVM model with variance-based feature selection
    print("\nTraining SVM model with variance-based feature selection...")
    var_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    var_model.fit(X_train_var, y_train)
    
    # Evaluate the variance-based model
    var_accuracy = evaluate_model(var_model, X_test_var, y_test, "SVM with Variance-based Feature Selection")
    
    # Apply RFE for feature selection
    # We'll select a reasonable number of features (you can adjust this)
    n_features_to_select = X_train_scaled.shape[1] // 2
    X_train_rfe, X_test_rfe = rfe_feature_selection(X_train_scaled, X_test_scaled, y_train, n_features_to_select)
    
    # Train the SVM model with RFE feature selection
    print("\nTraining SVM model with RFE feature selection...")
    rfe_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    rfe_model.fit(X_train_rfe, y_train)
    
    # Evaluate the RFE-based model
    rfe_accuracy = evaluate_model(rfe_model, X_test_rfe, y_test, "SVM with RFE Feature Selection")

    # SelectKBest Feature Selection (ANOVA F-Value)
    from sklearn.feature_selection import SelectKBest, f_classif
    
    print("\nApplying SelectKBest Feature Selection (Top 20 Features)...")

    selector_kbest = SelectKBest(score_func=f_classif, k=20)
    X_train_kbest = selector_kbest.fit_transform(X_train, y_train)
    X_test_kbest = selector_kbest.transform(X_test)
    
    # Retrain SVM on top-K features
    kbest_model = SVC()
    kbest_model.fit(X_train_kbest, y_train)
    y_pred_kbest = kbest_model.predict(X_test_kbest)
    kbest_accuracy = accuracy_score(y_test, y_pred_kbest)
    print("SelectKBest (Top 20) Accuracy:", kbest_accuracy)
    
    # Compare the results
    print("\nComparison of Feature Selection Methods:")
    print(f"Baseline SVM Accuracy: {baseline_accuracy:.4f}")
    print(f"Variance-based Feature Selection Accuracy: {var_accuracy:.4f}")
    print(f"RFE Feature Selection Accuracy: {rfe_accuracy:.4f}")
    
    # Plot the comparison
    methods = ['Baseline', 'Variance-based', 'RFE']
    accuracies = [baseline_accuracy, var_accuracy, rfe_accuracy]
    
    plt.figure(figsize=(10, 6))
    plt.bar(methods, accuracies, color=['blue', 'green', 'orange'])
    plt.ylim(0, 1.0)
    plt.xlabel('Feature Selection Method')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Feature Selection Methods')
    plt.savefig('feature_selection_comparison.png')
    print("\nComparison plot saved as 'feature_selection_comparison.png'")

if __name__ == "__main__":
    main()
