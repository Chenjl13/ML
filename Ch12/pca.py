'''
Created on Jun 1, 2011
Principal Component Analysis (PCA) for Dimensionality Reduction
Handles missing values (NaN) by replacing with feature mean
@author: Peter Harrington
Python3 Compatible Version (Fixed: map iterator, file encoding, complex eigenvalue handling)
'''
import numpy as np

def loadDataSet(fileName, delim='\t'):
    """
    Load dataset from text file and convert to numpy matrix of floats.
    Fixed: Convert map iterator to list for Python3 compatibility.
    Args:
        fileName (str): Path to the dataset file
        delim (str): Delimiter for parsing (default: tab '\t')
    Returns:
        np.matrix: Dataset matrix (rows = samples, columns = features)
    """
    # Fixed: Add encoding='utf-8' to avoid file read errors
    with open(fileName, encoding='utf-8') as fr:
        # Split each line by delimiter and strip whitespace
        stringArr = [line.strip().split(delim) for line in fr.readlines()]
        # Fixed: map() returns iterator in Python3 → convert to list
        datArr = [list(map(float, line)) for line in stringArr]
        # Convert list to numpy matrix
        return np.mat(datArr)

def pca(dataMat, topNfeat=9999999):
    """
    Perform Principal Component Analysis (PCA) for dimensionality reduction.
    Args:
        dataMat (np.matrix): Input dataset matrix (rows = samples, columns = features)
        topNfeat (int): Number of top principal components to retain (default: all features)
    Returns:
        tuple: (lowDDataMat, reconMat)
            lowDDataMat: Reduced-dimensional dataset (rows = samples, columns = topNfeat)
            reconMat: Reconstructed dataset in original dimension (for error evaluation)
    """
    # Step 1: Center the data (subtract feature mean)
    meanVals = np.mean(dataMat, axis=0)  # Mean of each feature (1 x n matrix)
    meanRemoved = dataMat - meanVals     # Centered data (m x n matrix)
    
    # Step 2: Compute covariance matrix (n x n matrix)
    # rowvar=0: each column is a feature (default: row is feature)
    covMat = np.cov(meanRemoved, rowvar=0)
    
    # Step 3: Solve eigenvalues and eigenvectors of covariance matrix
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    
    # Fixed: Handle complex eigenvalues (due to numerical precision) by taking real parts
    eigVals = eigVals.real
    eigVects = eigVects.real
    
    # Step 4: Sort eigenvalues in ascending order → reverse to get top N
    eigValInd = np.argsort(eigVals)  # Indices of sorted eigenvalues (smallest → largest)
    # Select top N eigenvalues (slice from end to topNfeat)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVects[:, eigValInd]  # Top N eigenvectors (n x topNfeat matrix)
    
    # Step 5: Project centered data to low-dimensional space (m x topNfeat matrix)
    lowDDataMat = meanRemoved * redEigVects
    
    # Step 6: Reconstruct data to original dimension (for comparison)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    
    return lowDDataMat, reconMat

def replaceNanWithMean():
    """
    Load 'secom.data' dataset and replace NaN values with the mean of the corresponding feature.
    Returns:
        np.matrix: Dataset with NaN replaced by feature means
    """
    # Load dataset with space as delimiter
    datMat = loadDataSet('secom.data', ' ')
    numFeat = datMat.shape[1]  # Number of features (columns)
    
    for i in range(numFeat):
        # Get indices of non-NaN values for current feature
        # ~isnan: invert NaN mask; nonzero: get non-zero indices (valid values)
        validInd = np.nonzero(~np.isnan(datMat[:, i].A))[0]
        # Compute mean of valid values (avoid NaN in mean calculation)
        meanVal = np.mean(datMat[validInd, i])
        # Replace NaN values with feature mean
        nanInd = np.nonzero(np.isnan(datMat[:, i].A))[0]
        if len(nanInd) > 0:
            datMat[nanInd, i] = meanVal
    
    return datMat

# ------------------------------
# Example Usage (Runs automatically when executing the script)
# ------------------------------
if __name__ == "__main__":
    # 1. Test PCA with sample data (optional: create your own sample or use secom.data)
    print("="*50)
    print("Test 1: PCA with Simulated High-Dimensional Data")
    print("="*50)
    # Create simulated data (100 samples, 5 features)
    np.random.seed(42)  # For reproducibility
    simulated_data = np.mat(np.random.randn(100, 5))  # 100x5 matrix
    print(f"Original data shape: {simulated_data.shape}")
    
    # Reduce dimension from 5 to 2
    topNfeat = 2
    lowDData, reconData = pca(simulated_data, topNfeat)
    print(f"Reduced data shape: {lowDData.shape}")
    print(f"Reconstructed data shape (original dimension): {reconData.shape}")
    
    # Calculate reconstruction error (mean squared error)
    reconError = np.mean(np.power(simulated_data - reconData, 2))
    print(f"Reconstruction MSE (lower = better): {reconError:.6f}")
    
    # 2. Test PCA with real dataset (secom.data, handle missing values first)
    print("\n" + "="*50)
    print("Test 2: PCA with Secom Dataset (Handle Missing Values)")
    print("="*50)
    try:
        # Load dataset and replace NaN with mean
        secom_data = replaceNanWithMean()
        print(f"Secom dataset shape (after NaN handling): {secom_data.shape}")
        
        # Reduce dimension from original features to 3
        topNfeat_secom = 3
        lowD_secom, recon_secom = pca(secom_data, topNfeat_secom)
        print(f"Secom data shape after PCA (3 components): {lowD_secom.shape}")
        
        # Reconstruction error
        secom_recon_error = np.mean(np.power(secom_data - recon_secom, 2))
        print(f"Secom reconstruction MSE: {secom_recon_error:.6f}")
    except FileNotFoundError:
        print("Warning: 'secom.data' not found. Skipping real dataset test.")
        print("Download 'secom.data' from UCI Machine Learning Repository and place in the same folder.")