'''
Created on Mar 8, 2011
Collaborative Filtering Recommendation System + SVD Applications
- Personalized item recommendation (user-item rating matrix)
- Image compression using SVD dimensionality reduction
@author: Peter Harrington
Python3 Compatible Version (Fixed: print syntax, file encoding, robust error handling)
'''
import numpy as np
from numpy import linalg as la

def loadExData():
    """Load sample user-item rating matrix (0 = unrated)"""
    return [
        [0, 0, 0, 2, 2],
        [0, 0, 0, 3, 3],
        [0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0],
        [2, 2, 2, 0, 0],
        [5, 5, 5, 0, 0],
        [1, 1, 1, 0, 0]
    ]
    
def loadExData2():
    """Load extended sample user-item rating matrix (0 = unrated)"""
    return [
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
        [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
        [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
        [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
        [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
        [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
        [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
        [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
        [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]
    ]
    
def ecludSim(inA, inB):
    """
    Calculate Euclidean distance-based similarity (normalized to 0~1).
    Higher value means more similar.
    Args:
        inA (np.matrix): Vector A (item/user features)
        inB (np.matrix): Vector B (item/user features)
    Returns:
        float: Similarity score (0~1)
    """
    return 1.0 / (1.0 + la.norm(inA - inB))

def pearsSim(inA, inB):
    """
    Calculate Pearson correlation coefficient-based similarity (normalized to 0~1).
    Robust to extreme values; higher value means more similar.
    Args:
        inA (np.matrix): Vector A (item/user features)
        inB (np.matrix): Vector B (item/user features)
    Returns:
        float: Similarity score (0~1)
    """
    if len(inA) < 3:
        return 1.0  # Not enough data → full similarity
    # Normalize to 0~1 (Pearson correlation ranges from -1 to 1)
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]

def cosSim(inA, inB):
    """
    Calculate cosine similarity (normalized to 0~1).
    Suitable for sparse data; higher value means more similar.
    Args:
        inA (np.matrix): Vector A (item/user features)
        inB (np.matrix): Vector B (item/user features)
    Returns:
        float: Similarity score (0~1)
    """
    num = float(inA.T * inB)  # Dot product
    denom = la.norm(inA) * la.norm(inB)  # Product of magnitudes
    # Avoid division by zero; normalize to 0~1 (cosine ranges from -1 to 1)
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0.0

def standEst(dataMat, user, simMeas, item):
    """
    Standard rating estimation: Predict user's rating for item using item-based similarity.
    Args:
        dataMat (np.matrix): User-item rating matrix (rows=users, cols=items)
        user (int): Target user index
        simMeas (function): Similarity measurement function
        item (int): Target item index (unrated by user)
    Returns:
        float: Estimated rating for the item
    """
    n = dataMat.shape[1]  # Number of items
    simTotal = 0.0
    ratSimTotal = 0.0
    
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue  # Skip unrated items
        
        # Find users who rated both target item (item) and current item (j)
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        
        if len(overLap) == 0:
            similarity = 0.0
        else:
            # Calculate similarity between target item and current item
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        
        print(f"Item {item} and Item {j} similarity: {similarity:.4f}")
        simTotal += similarity
        ratSimTotal += similarity * userRating  # Weighted sum of ratings
    
    return ratSimTotal / simTotal if simTotal != 0 else 0.0

def svdEst(dataMat, user, simMeas, item):
    """
    SVD-optimized rating estimation: Reduce dimension via SVD before calculating similarity.
    Reduces computation and improves performance for sparse data.
    Args:
        dataMat (np.matrix): User-item rating matrix (rows=users, cols=items)
        user (int): Target user index
        simMeas (function): Similarity measurement function
        item (int): Target item index (unrated by user)
    Returns:
        float: Estimated rating for the item
    """
    n = dataMat.shape[1]  # Number of items
    simTotal = 0.0
    ratSimTotal = 0.0
    
    # Perform SVD on rating matrix (U: user feature, Sigma: singular values, VT: item feature)
    U, Sigma, VT = la.svd(dataMat)
    # Retain top 4 singular values (dimensionality reduction to 4)
    Sig4 = np.mat(np.eye(4) * Sigma[:4])
    # Transform items to low-dimensional space (cols=items → rows=items)
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue  # Skip unrated items or target item
        
        # Calculate similarity in low-dimensional space
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print(f"Item {item} and Item {j} similarity (SVD-optimized): {similarity:.4f}")
        
        simTotal += similarity
        ratSimTotal += similarity * userRating  # Weighted sum of ratings
    
    return ratSimTotal / simTotal if simTotal != 0 else 0.0

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    """
    Recommend top-N unrated items for target user based on rating estimation.
    Args:
        dataMat (np.matrix): User-item rating matrix (rows=users, cols=items)
        user (int): Target user index
        N (int): Number of recommendations to return (default: 3)
        simMeas (function): Similarity measurement function (default: cosSim)
        estMethod (function): Rating estimation method (default: standEst)
    Returns:
        list/tuple: Top-N (item index, estimated rating) or message if all items are rated
    """
    # Find indices of unrated items for target user
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    
    if len(unratedItems) == 0:
        return "You have rated all items!"
    
    itemScores = []
    for item in unratedItems:
        # Estimate rating for unrated item
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    
    # Sort by estimated rating (descending) and return top-N
    return sorted(itemScores, key=lambda x: x[1], reverse=True)[:N]

def printMat(inMat, thresh=0.8):
    """
    Print binary image matrix (32x32) for visualization (1=foreground, 0=background).
    Args:
        inMat (np.matrix): Image matrix (32x32)
        thresh (float): Threshold to binarize values (default: 0.8)
    """
    for i in range(32):
        for k in range(32):
            # Binarize: value > thresh → 1, else 0
            print(1 if float(inMat[i, k]) > thresh else 0, end=' ')
        print()  # New line after each row

def imgCompress(numSV=3, thresh=0.8):
    """
    Compress 32x32 binary image using SVD (retain top-N singular values).
    Args:
        numSV (int): Number of singular values to retain (default: 3)
        thresh (float): Threshold for image binarization (default: 0.8)
    """
    # Load 32x32 binary image from file (each line has 32 characters: 0/1)
    myl = []
    try:
        # Fixed: Add encoding='utf-8' to avoid file read errors
        with open('0_5.txt', encoding='utf-8') as fr:
            for line in fr.readlines():
                line = line.strip()
                if len(line) != 32:
                    continue  # Skip invalid lines
                newRow = [int(c) for c in line]
                myl.append(newRow)
    except FileNotFoundError:
        print("Error: '0_5.txt' not found. Cannot perform image compression.")
        print("Download '0_5.txt' (32x32 binary image file) and place in the same folder.")
        return
    
    myMat = np.mat(myl)
    print("**** Original Image Matrix ****")
    printMat(myMat, thresh)
    
    # Perform SVD on image matrix
    U, Sigma, VT = la.svd(myMat)
    # Reconstruct sigma matrix with top-N singular values
    SigRecon = np.mat(np.zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    
    # Reconstruct image matrix from top-N singular values
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print(f"\n**** Reconstructed Image (using {numSV} singular values) ****")
    printMat(reconMat, thresh)

# ------------------------------
# Example Usage (Runs automatically when executing the script)
# ------------------------------
if __name__ == "__main__":
    # Convert sample data to numpy matrix
    dataMat = np.mat(loadExData2())
    
    # 1. Test Recommendation System (Target User = 2)
    print("="*50)
    print("Test 1: Personalized Item Recommendation")
    print("="*50)
    targetUser = 2
    print(f"Recommending items for User {targetUser} (standard estimation + cosine similarity):")
    recs_stand = recommend(dataMat, targetUser, N=3, simMeas=cosSim, estMethod=standEst)
    print(f"Top 3 Recommendations: {recs_stand}\n")
    
    print(f"Recommending items for User {targetUser} (SVD-optimized + Pearson similarity):")
    recs_svd = recommend(dataMat, targetUser, N=3, simMeas=pearsSim, estMethod=svdEst)
    print(f"Top 3 Recommendations: {recs_svd}\n")
    
    # 2. Test Image Compression (Uncomment to use, requires '0_5.txt')
    """
    print("="*50)
    print("Test 2: Image Compression with SVD")
    print("="*50)
    imgCompress(numSV=3)  # Retain 3 singular values (compression ratio ~ 32*32/(32*3 + 3 + 3*32) ≈ 10x)
    """