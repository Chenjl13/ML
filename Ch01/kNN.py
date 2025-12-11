import numpy as np
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    """
    Core kNN classification function.
    Args:
        inX (np.ndarray): Test vector (1xN)
        dataSet (np.ndarray): Training dataset (MxN, M=number of samples, N=number of features)
        labels (list): Label vector for training dataset (1xM)
        k (int): Number of nearest neighbors to use (preferably odd)
    Returns:
        str/int: Most frequent class label (prediction for inX)
    """
    dataSetSize = dataSet.shape[0]
    # Calculate Euclidean distance between inX and all training samples
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # Repeat inX to match dataSet shape
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # Sum along rows (get squared distance for each sample)
    distances = sqDistances ** 0.5  # Euclidean distance
    
    # Sort distances and get indices of sorted values
    sortedDistIndicies = distances.argsort()
    
    # Vote for the most frequent label
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # Count label frequency
    
    # Sort classCount by frequency (descending)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    """Generate sample dataset for kNN testing"""
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def file2matrix(filename):
    """
    Convert dating dataset file to feature matrix and label vector.
    Args:
        filename (str): Path to dating dataset file
    Returns:
        tuple: (feature_matrix, label_vector)
            feature_matrix (np.ndarray): Mx3 matrix (M=samples, 3 features)
            label_vector (list): 1xM list of labels (1/2/3)
    """
    # Fixed: Add encoding='utf-8' to avoid file read errors
    with open(filename, encoding='utf-8') as fr:
        numberOfLines = len(fr.readlines())  # Get total number of samples
    
    # Initialize feature matrix (Mx3) and label vector
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    
    # Read file again to extract data (file pointer reset)
    with open(filename, encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip()  # Remove whitespace and newline characters
            listFromLine = line.split('\t')  # Split by tab (dataset delimiter)
            returnMat[index, :] = listFromLine[0:3]  # First 3 elements = features
            classLabelVector.append(int(listFromLine[-1]))  # Last element = label
            index += 1
    
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
    Normalize feature values to [0, 1] (min-max normalization).
    Args:
        dataSet (np.ndarray): Feature matrix (MxN)
    Returns:
        tuple: (normDataSet, ranges, minVals)
            normDataSet (np.ndarray): Normalized feature matrix
            ranges (np.ndarray): Range of each feature (max - min)
            minVals (np.ndarray): Minimum value of each feature
    """
    minVals = dataSet.min(0)  # Minimum value per feature (1xN)
    maxVals = dataSet.max(0)  # Maximum value per feature (1xN)
    ranges = maxVals - minVals  # Range of each feature (1xN)
    
    m = dataSet.shape[0]
    # Normalization formula: (x - min) / (max - min)
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # Subtract min from all samples
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # Divide by range (element-wise)
    
    return normDataSet, ranges, minVals

def datingClassTest():
    """Test kNN on dating dataset: Calculate classification error rate"""
    hoRatio = 0.50  # Hold out 50% of data as test set (adjustable)
    # Load dataset and normalize features
    try:
        datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    except FileNotFoundError:
        print("Error: 'datingTestSet2.txt' not found. Please place the file in the same folder.")
        return
    
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)  # Number of test samples
    errorCount = 0.0
    
    # Classify each test sample
    for i in range(numTestVecs):
        # Use latter 50% as training set, former 50% as test set
        classifierResult = classify0(
            normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], k=3
        )
        # Fixed: Python2 print → Python3 print()
        print(f"Classifier prediction: {classifierResult}, Actual label: {datingLabels[i]}")
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    
    # Print error rate
    errorRate = errorCount / float(numTestVecs)
    print(f"\nTotal error rate: {errorRate:.4f}")
    print(f"Total number of errors: {int(errorCount)}")

def img2vector(filename):
    """
    Convert 32x32 binary image file to 1x1024 feature vector.
    Args:
        filename (str): Path to 32x32 binary image file
    Returns:
        np.ndarray: 1x1024 feature vector
    """
    returnVect = np.zeros((1, 1024))  # 32*32=1024
    # Fixed: Add encoding='utf-8' to avoid file read errors
    with open(filename, encoding='utf-8') as fr:
        for i in range(32):
            lineStr = fr.readline().strip()  # Read each line (32 characters)
            for j in range(32):
                returnVect[0, 32*i + j] = int(lineStr[j])  # Assign pixel value to vector
    
    return returnVect

def handwritingClassTest():
    """Test kNN on handwritten digit dataset: Calculate classification error rate"""
    hwLabels = []
    # Load training set (from 'trainingDigits' folder)
    try:
        trainingFileList = listdir('trainingDigits')
    except FileNotFoundError:
        print("Error: 'trainingDigits' folder not found. Please place the folder in the same directory.")
        return
    
    m = len(trainingFileList)  # Number of training samples
    trainingMat = np.zeros((m, 1024))  # Training feature matrix (Mx1024)
    
    # Extract training data and labels
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # Remove file extension (e.g., "1_0.txt" → "1_0")
        classNumStr = int(fileStr.split('_')[0])  # Extract label (e.g., "1_0" → 1)
        hwLabels.append(classNumStr)
        # Convert image to vector and store in trainingMat
        trainingMat[i, :] = img2vector(f'trainingDigits/{fileNameStr}')
    
    # Load test set (from 'testDigits' folder)
    try:
        testFileList = listdir('testDigits')
    except FileNotFoundError:
        print("Error: 'testDigits' folder not found. Please place the folder in the same directory.")
        return
    
    errorCount = 0.0
    mTest = len(testFileList)  # Number of test samples
    
    # Classify each test sample
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])  # Actual label
        # Convert test image to vector
        vectorUnderTest = img2vector(f'testDigits/{fileNameStr}')
        # kNN classification (k=3)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, k=3)
        # Fixed: Python2 print → Python3 print()
        print(f"Classifier prediction: {classifierResult}, Actual label: {classNumStr}")
        if classifierResult != classNumStr:
            errorCount += 1.0
    
    # Print test results
    print(f"\nTotal number of errors: {int(errorCount)}")
    errorRate = errorCount / float(mTest)
    print(f"Total error rate: {errorRate:.4f}")

# ------------------------------
# Example Usage (Uncomment to run)
# ------------------------------
if __name__ == "__main__":
    # Test 1: Run dating website classification test
    # datingClassTest()
    
    # Test 2: Run handwritten digit recognition test
    # handwritingClassTest()
    
    pass