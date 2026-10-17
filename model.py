"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    clean = np.copy(X)
    means = np.nanmean(clean, axis = 0)
    means [np.isnan(means)] = 0 
    missing = np.isnan(clean)  
    for j in range(clean.shape[1]):
        clean[missing [:, j], j] = means [j]
    return clean 


    pass

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1 = np.percentile(X,25, axis = 0)
    q3 = np.percentile(X, 75, axis = 0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper
    
    
    pass

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    clean = np.copy(X)
    for j in range (clean.shape[1]):
        clean[:, j] = np.clip(clean[:, j], lower[j], upper[j])
    return clean 
    
    pass

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    
    feature = numerator / (denominator + eps)
    return feature 
    
    
    pass

# Step 5 - append_column
def append_column(X, col):
# TODO: Horizontally append one 1-D feature column onto a design matrix.

    X_new = np.append(X,  col.reshape(-1,1), axis = 1) 
    return X_new

# Step 6 - one_hot_encode
import numpy as np

def one_hot_encode(labels):
    # Find the unique categories
    categories = np.unique(labels)

    # Number of rows and columns
    N = len(labels)
    C = len(categories)

    # Create an empty matrix of zeros
    encoded = np.zeros((N, C))

    # Go through each label
    for i, label in enumerate(labels):

        # Find which column belongs to this label
        j = np.where(categories == label)[0][0]

        # Put a 1 in that position
        encoded[i, j] = 1

    return encoded

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
   
   mean = np.mean (X, axis = 0 )
   std = np.std (X, axis = 0)
   std = np.where(std == 0, 1.0, std)

   return (mean, std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    
    scaled_matrix = (X - mean)/ std 
    
    return scaled_matrix
    
    pass

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    ones = np.ones((X.shape[0],1))

    bias_matrix = np.hstack((ones, X))
    return bias_matrix
    pass

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    rng = np.random.default_rng(seed) 
    idx = rng.permutation (n_samples)

    return idx 
    
    pass

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
   
   n = len(indices)
   n_train = int(n*train_ratio)
   n_val = int(n * val_ratio)
   

   train_idx = indices[:n_train]
   val_idx = indices[n_train: n_val+n_train]
   test_idx = indices[n_val+n_train :]

   return train_idx,val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    
    X_sub = X[indices]
    y_sub = y[indices]

    return X_sub, y_sub
    
    
    pass

# Step 13 - ols_fit (not yet solved)
# TODO: implement

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

