'''
Created on Feb 4, 2011
Tree-Based Regression Methods (Python3 Compatible)
@author: Peter Harrington
'''
import numpy as np

def loadDataSet(fileName):
    """
    Load tab-delimited dataset. Last column is the continuous target value.
    Args:
        fileName (str): Path to the dataset file
    Returns:
        list: 2D list of [features, target], all values are floats
    """
    dataMat = []
    fr = open(fileName, encoding='utf-8')
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # Convert map object to list (Python3 compatibility fix)
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    fr.close()
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    """
    Split dataset based on a feature and threshold value.
    Handles empty index cases to avoid index out-of-bounds errors.
    Args:
        dataSet (np.matrix): Input dataset (matrix format)
        feature (int): Index of the feature to split on
        value (float): Threshold value for splitting
    Returns:
        tuple: Two sub-matrices (data where feature > value, data where feature <= value)
    """
    idx0 = np.nonzero(dataSet[:, feature] > value)[0]
    idx1 = np.nonzero(dataSet[:, feature] <= value)[0]
    mat0 = dataSet[idx0, :] if len(idx0) > 0 else np.mat([])
    mat1 = dataSet[idx1, :] if len(idx1) > 0 else np.mat([])
    return mat0, mat1

def regLeaf(dataSet):
    """
    Leaf node for regression tree: returns the mean of target values in the subset.
    Args:
        dataSet (np.matrix): Subset of data at current node
    Returns:
        float: Mean of the target values
    """
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    """
    Error calculation for regression tree: variance * number of samples.
    Args:
        dataSet (np.matrix): Subset of data
    Returns:
        float: Total error of the subset
    """
    return np.var(dataSet[:, -1]) * dataSet.shape[0]

def linearSolve(dataSet):
    """
    Helper function for model tree: solve linear regression parameters (w).
    Args:
        dataSet (np.matrix): Subset of data for linear fitting
    Returns:
        tuple: (weights, X_matrix, Y_vector)
    Raises:
        NameError: If matrix is singular (non-invertible)
    """
    m, n = dataSet.shape
    X = np.mat(np.ones((m, n)))  # Add bias term (1st column)
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]  # Features (exclude target)
    Y = dataSet[:, -1]  # Target values
    xTx = X.T @ X  # Matrix multiplication (Python3 recommended syntax)
    
    # Check if matrix is invertible
    if np.linalg.det(xTx) == 0.0:
        raise NameError('Matrix is singular, cannot compute inverse.\nTry increasing the second value of ops (e.g., set to 20)')
    
    ws = xTx.I @ (X.T @ Y)  # Solve for weights
    return ws, X, Y

def modelLeaf(dataSet):
    """
    Leaf node for model tree: returns trained linear model parameters.
    Args:
        dataSet (np.matrix): Subset of data for linear fitting
    Returns:
        np.matrix: Weights of the linear model
    """
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    """
    Error calculation for model tree: sum of squared residuals.
    Args:
        dataSet (np.matrix): Subset of data
    Returns:
        float: Total squared error of the linear model
    """
    ws, X, Y = linearSolve(dataSet)
    yHat = X @ ws  # Predictions from linear model
    return np.sum(np.power(Y - yHat, 2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    Find the best feature and threshold to split the data (minimizes error).
    Args:
        dataSet (np.matrix): Input dataset
        leafType (function): Function to compute leaf node value
        errType (function): Function to compute error
        ops (tuple): (tolS, tolN) - tolerance for error reduction and min sample count
    Returns:
        tuple: (best_feature_index, best_threshold_value) or (None, leaf_value) if split stops
    """
    tolS = ops[0]  # Minimum reduction in error to allow split
    tolN = ops[1]  # Minimum number of samples required for split
    
    # Stop condition 1: All target values are the same
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    
    m, n = dataSet.shape
    baseErr = errType(dataSet)
    bestErr = float('inf')
    bestFeat = 0
    bestVal = 0
    
    # Iterate all features (exclude target column)
    for featIndex in range(n - 1):
        # Iterate all unique values of current feature (convert matrix to 1D array first)
        for splitVal in set(dataSet[:, featIndex].A.flatten()):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            
            # Skip if either subset is too small
            if mat0.shape[0] < tolN or mat1.shape[0] < tolN:
                continue
            
            # Calculate total error after split
            newErr = errType(mat0) + errType(mat1)
            
            # Update best split if error is reduced
            if newErr < bestErr:
                bestFeat = featIndex
                bestVal = splitVal
                bestErr = newErr
    
    # Stop condition 2: Error reduction is insufficient
    if (baseErr - bestErr) < tolS:
        return None, leafType(dataSet)
    
    # Stop condition 3: Split results in too small subsets
    mat0, mat1 = binSplitDataSet(dataSet, bestFeat, bestVal)
    if mat0.shape[0] < tolN or mat1.shape[0] < tolN:
        return None, leafType(dataSet)
    
    return bestFeat, bestVal

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    Recursively build the regression tree/model tree.
    Args:
        dataSet (np.matrix): Input dataset
        leafType (function): Function to compute leaf node value
        errType (function): Function to compute error
        ops (tuple): Tolerance parameters for splitting
    Returns:
        dict/float: Tree structure (dict) or leaf node value (float)
    """
    # Get best split or leaf value
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    
    # If no valid split, return leaf value
    if feat is None:
        return val
    
    # Build tree structure
    tree = {}
    tree['spInd'] = feat  # Split feature index
    tree['spVal'] = val   # Split threshold value
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    
    # Recursively build left and right subtrees
    tree['left'] = createTree(lSet, leafType, errType, ops)
    tree['right'] = createTree(rSet, leafType, errType, ops)
    
    return tree

def isTree(obj):
    """
    Check if the input object is a tree (dict structure) or a leaf value.
    Args:
        obj: Input object to check
    Returns:
        bool: True if obj is a tree (dict), False otherwise
    """
    return type(obj).__name__ == 'dict'

def getMean(tree):
    """
    Compute the mean value of all leaf nodes in a tree (used for pruning).
    Args:
        tree (dict/float): Tree structure or leaf value
    Returns:
        float: Mean of all leaf values
    """
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    """
    Prune the tree using test data to avoid overfitting.
    Merges branches if error is reduced.
    Args:
        tree (dict/float): Trained tree structure
        testData (np.matrix): Test dataset for pruning
    Returns:
        dict/float: Pruned tree or merged leaf value
    """
    # If no test data, return mean of leaves
    if testData.shape[0] == 0:
        return getMean(tree)
    
    # Split test data using the tree's split rule
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    
    # Recursively prune left and right subtrees
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    
    # Merge branches if both are leaves
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        
        # Calculate error without merging
        errNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                     np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        
        # Calculate error with merging (use mean of leaves)
        treeMean = (tree['left'] + tree['right']) / 2.0
        errMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        
        # Merge if error is reduced
        if errMerge < errNoMerge:
            print("Merging branches (pruning)")
            return treeMean
        else:
            return tree
    else:
        return tree

def regTreeEval(model, inDat):
    """
    Evaluate prediction for regression tree (leaf value is mean).
    Args:
        model (float): Leaf value of regression tree
        inDat (np.matrix): Single input sample
    Returns:
        float: Prediction
    """
    return float(model)

def modelTreeEval(model, inDat):
    """
    Evaluate prediction for model tree (leaf value is linear model).
    Args:
        model (np.matrix): Weights of the linear model
        inDat (np.matrix): Single input sample
    Returns:
        float: Prediction from linear model
    """
    n = inDat.shape[1]
    X = np.mat(np.ones((1, n + 1)))  # Add bias term
    X[:, 1:n+1] = inDat
    return float(X @ model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    """
    Make prediction for a single input sample using the trained tree.
    Args:
        tree (dict/float): Trained tree structure
        inData (np.matrix): Single input sample
        modelEval (function): Evaluation function (regTreeEval/modelTreeEval)
    Returns:
        float: Prediction
    """
    # If leaf node, return evaluation
    if not isTree(tree):
        return modelEval(tree, inData)
    
    # Traverse tree based on feature value
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    """
    Make batch predictions for multiple test samples.
    Args:
        tree (dict/float): Trained tree structure
        testData (list): List of test samples
        modelEval (function): Evaluation function (regTreeEval/modelTreeEval)
    Returns:
        np.matrix: Predictions (m x 1 matrix, m = number of test samples)
    """
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))  # Store predictions
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat