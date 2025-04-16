"""
Let's tackle the MEU Mobile KSD Dataset using variance-based feature selection and RFE

This script does a few key things:
1. Loads up the keystrokes dataset and gets it ready for analysis
2. Tries out a vanilla SVM model as our baseline
3. Applies variance-based feature selection to see if we can simplify things
4. Tests out RFE as another way to find the most important features
5. Compares how well each approach performs
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
    See how well our model is doing
    
    Args:
        model: Our trained classifier
        X_test: Test data features
        y_test: The actual user IDs we want to predict
        title: Name for this evaluation section
    
    Returns:
        accuracy: How often the model is right
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{title}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nDetailed breakdown by class:")
    print(classification_report(y_test, y_pred))
    
    return accuracy

def variance_based_feature_selection(X_train, X_test, threshold=0.0):
    """
    Get rid of features that don't vary much across samples
    
    Args:
        X_train: Training data
        X_test: Test data
        threshold: Features with variance below this get dropped
        
    Returns:
        X_train_var, X_test_var: Datasets with only the high-variance features
    """
    # Create our feature filter
    selector = VarianceThreshold(threshold=threshold)
    
    # Apply the filter to our training data
    X_train_var = selector.fit_transform(X_train)
    
    # Use the same filter on our test data
    X_test_var = selector.transform(X_test)
    
    # Find out which features made the cut
    selected_features = selector.get_support(indices=True)
    
    print(f"\nVariance-based Feature Selection (threshold={threshold})")
    print(f"Number of features before selection: {X_train.shape[1]}")
    print(f"Number of features after selection: {X_train_var.shape[1]}")
    
    # Get the names of our selected features
    selected_feature_names = [X_train.columns[i] for i in selected_features]
    
    if len(selected_features) < X_train.shape[1]:
        print(f"Features removed: {X_train.shape[1] - len(selected_features)}")
        
        # Show some of the features we kept
        print("\nSelected features (just showing the first 10):")
        for i, feature in enumerate(selected_feature_names[:10]):
            print(f"  {i+1}. {feature}")
        
        if len(selected_feature_names) > 10:
            print(f"  ... and {len(selected_feature_names) - 10} more features")
    else:
        print("No features were removed based on the variance threshold.")
    
    # Package everything back into DataFrames with proper column names
    X_train_var_df = pd.DataFrame(X_train_var, columns=selected_feature_names)
    X_test_var_df = pd.DataFrame(X_test_var, columns=selected_feature_names)
    
    return X_train_var_df, X_test_var_df

def rfe_feature_selection(X_train, X_test, y_train, n_features_to_select=None):
    """
    Let's be more selective - find the most important features by training and ranking
    
    Args:
        X_train: Training data
        X_test: Test data
        y_train: Target values for training
        n_features_to_select: How many features to keep
        
    Returns:
        X_train_rfe, X_test_rfe: Datasets with only the best features
    """
    # If not specified, let's keep half of the features
    if n_features_to_select is None:
        n_features_to_select = X_train.shape[1] // 2
    
    # We'll use a linear SVM to help us rank features
    svm = SVC(kernel="linear", random_state=42)
    
    # Set up RFE to recursively eliminate features
    selector = RFE(estimator=svm, n_features_to_select=n_features_to_select, step=1)
    
    # Train it on our data to find the best features
    X_train_rfe = selector.fit_transform(X_train, y_train)
    
    # Apply the same transformation to test data
    X_test_rfe = selector.transform(X_test)
    
    # Get the indices of features we're keeping
    selected_features = np.where(selector.support_)[0]
    
    print(f"\nRecursive Feature Elimination (keeping {n_features_to_select} features)")
    print(f"Number of features before selection: {X_train.shape[1]}")
    print(f"Number of features after selection: {X_train_rfe.shape[1]}")
    
    # Let's see which features made the cut
    selected_feature_names = [X_train.columns[i] for i in selected_features]
    print("\nTop features (just showing the first 10):")
    for i, feature in enumerate(selected_feature_names[:10]):
        print(f"  {i+1}. {feature}")
    
    if len(selected_feature_names) > 10:
        print(f"  ... and {len(selected_feature_names) - 10} more features")
    
    # Show the features ranked by importance
    feature_ranks = {str(col): rank for col, rank in zip(X_train.columns, selector.ranking_)}
    sorted_features = sorted(feature_ranks.items(), key=lambda x: x[1])
    print("\nTop 10 features by importance (lower rank = more important):")
    for i, (feature, rank) in enumerate(sorted_features[:10]):
        print(f"  {i+1}. {feature}: {rank}")
    
    # Package everything back into DataFrames with proper column names
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
