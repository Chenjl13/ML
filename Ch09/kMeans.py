'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
Python3 Compatible Version (Fixed: map iterator, urllib module, encoding issues)
'''
import numpy as np
import urllib.parse
import urllib.request
import json
from time import sleep
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    """
    General function to parse tab-delimited floats.
    Fixed: Convert map iterator to list (Python3 compatibility)
    Args:
        fileName (str): Path to the dataset file
    Returns:
        list: 2D list of float values
    """
    dataMat = []
    # Fixed: Add encoding='utf-8' to avoid file read errors
    with open(fileName, encoding='utf-8') as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            # Fixed: map() returns iterator in Python3, convert to list
            fltLine = list(map(float, curLine))
            dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    """Calculate Euclidean distance between two vectors"""
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))  # Equivalent to np.linalg.norm(vecA-vecB)

def randCent(dataSet, k):
    """
    Create random cluster centers within the range of each feature.
    Args:
        dataSet (np.matrix): Input dataset
        k (int): Number of clusters
    Returns:
        np.matrix: k cluster centers (k x n matrix, n = number of features)
    """
    n = dataSet.shape[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        rangeJ = float(np.max(dataSet[:, j]) - minJ)
        # Fixed: np.random.rand() is compatible with Python3
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    Basic k-Means clustering algorithm.
    Args:
        dataSet (np.matrix): Input dataset (m x n matrix)
        k (int): Number of clusters
        distMeas (function): Distance measurement function (default: Euclidean)
        createCent (function): Cluster center initialization function (default: random)
    Returns:
        tuple: (centroids, clusterAssment)
            centroids: Final cluster centers (k x n matrix)
            clusterAssment: Cluster assignment for each sample (m x 2 matrix: [cluster_id, squared_distance])
    """
    m = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = float('inf')
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # Update cluster if assignment changes
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        
        print(centroids)
        # Update cluster centers (mean of samples in each cluster)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            if ptsInClust.shape[0] > 0:  # Avoid empty cluster (divide by zero)
                centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    """
    Bisecting k-Means clustering (optimized version of basic k-Means).
    Solves local optimal problem by incremental splitting.
    Args:
        dataSet (np.matrix): Input dataset (m x n matrix)
        k (int): Number of clusters
        distMeas (function): Distance measurement function (default: Euclidean)
    Returns:
        tuple: (centroids, clusterAssment)
            centroids: Final cluster centers (k x n matrix)
            clusterAssment: Cluster assignment for each sample (m x 2 matrix: [cluster_id, squared_distance])
    """
    m = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    # Initialize with 1 cluster (mean of all samples)
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    
    # Calculate initial SSE (sum of squared errors)
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :]) ** 2
    
    # Split clusters until reaching k clusters
    while len(centList) < k:
        lowestSSE = float('inf')
        
        # Try splitting each existing cluster
        for i in range(len(centList)):
            # Extract samples in current cluster
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            # Split current cluster into 2 sub-clusters
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # Calculate SSE after splitting
            sseSplit = np.sum(splitClustAss[:, 1])
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            
            # Update best split if total SSE is lower
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        
        # Update cluster IDs for split cluster
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        
        # Update cluster centers
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        
        # Update cluster assignment for split cluster
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    
    return np.mat(centList), clusterAssment

def geoGrab(stAddress, city):
    """
    Get latitude and longitude from address using Yahoo Geocoding API (DEPRECATED).
    NOTE: Yahoo API is no longer available. Replace with Google/Amap/Baidu Geocoding API.
    Args:
        stAddress (str): Street address
        city (str): City name
    Returns:
        dict: Geocoding result (contains latitude/longitude if successful)
    """
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {
        'flags': 'J',
        'appid': 'aaa0VN6k',
        'location': '%s %s' % (stAddress, city)
    }
    # Fixed: urllib.urlencode → urllib.parse.urlencode (Python3)
    url_params = urllib.parse.urlencode(params)
    yahooApi = apiStem + url_params
    print(yahooApi)
    
    # Fixed: urllib.urlopen → urllib.request.urlopen (Python3)
    try:
        with urllib.request.urlopen(yahooApi) as c:
            # Fixed: Read response as string (Python3 returns bytes)
            response = c.read().decode('utf-8')
            return json.loads(response)
    except Exception as e:
        print("Error fetching geocode:", str(e))
        return {'ResultSet': {'Error': 1}}

def massPlaceFind(fileName):
    """
    Batch convert addresses to latitude/longitude and save to 'places.txt'.
    Depends on geoGrab (Yahoo API, deprecated).
    Args:
        fileName (str): Path to address file (tab-delimited: name\tstreet\tcity)
    """
    # Fixed: Add encoding='utf-8' for file write
    with open('places.txt', 'w', encoding='utf-8') as fw:
        # Fixed: Add encoding='utf-8' for file read
        with open(fileName, encoding='utf-8') as fr:
            for line in fr.readlines():
                line = line.strip()
                if not line:
                    continue
                lineArr = line.split('\t')
                if len(lineArr) < 3:
                    print("Invalid address line:", line)
                    continue
                retDict = geoGrab(lineArr[1], lineArr[2])
                if retDict['ResultSet']['Error'] == 0:
                    lat = float(retDict['ResultSet']['Results'][0]['latitude'])
                    lng = float(retDict['ResultSet']['Results'][0]['longitude'])
                    print("%s\t%f\t%f" % (lineArr[0], lat, lng))
                    fw.write('%s\t%f\t%f\n' % (line, lat, lng))
                else:
                    print("Error fetching geocode for:", line)
                sleep(1)  # Avoid API rate limit

def distSLC(vecA, vecB):
    """
    Calculate spherical distance between two points (latitude/longitude) on Earth.
    Args:
        vecA (np.matrix): [longitude, latitude] of point A
        vecB (np.matrix): [longitude, latitude] of point B
    Returns:
        float: Spherical distance in kilometers
    """
    # Convert degrees to radians
    latA = vecA[0, 1] * np.pi / 180.0
    latB = vecB[0, 1] * np.pi / 180.0
    lonA = vecA[0, 0] * np.pi / 180.0
    lonB = vecB[0, 0] * np.pi / 180.0
    
    a = np.sin(latA) * np.sin(latB)
    b = np.cos(latA) * np.cos(latB) * np.cos(lonB - lonA)
    # Earth radius: 6371 km
    return np.arccos(a + b) * 6371.0

def clusterClubs(numClust=5):
    """
    Cluster clubs based on geographic coordinates and visualize on a map.
    Requires 'places.txt' (generated by massPlaceFind) and 'Portland.png' (map image).
    Args:
        numClust (int): Number of clusters (default: 5)
    """
    datList = []
    # Read latitude/longitude from places.txt
    try:
        with open('places.txt', encoding='utf-8') as fr:
            for line in fr.readlines():
                line = line.strip()
                if not line:
                    continue
                lineArr = line.split('\t')
                if len(lineArr) < 5:
                    continue
                # Append [longitude, latitude]
                datList.append([float(lineArr[4]), float(lineArr[3])])
    except FileNotFoundError:
        print("Error: 'places.txt' not found. Run massPlaceFind first.")
        return
    
    datMat = np.mat(datList)
    # Use bisecting k-Means with spherical distance
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    
    # Plot visualization
    fig = plt.figure(figsize=(10, 8))
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    
    # Add map background
    ax0 = fig.add_axes(rect, label='ax0')
    try:
        imgP = plt.imread('Portland.png')
        ax0.imshow(imgP)
    except FileNotFoundError:
        print("Warning: 'Portland.png' not found. Skipping map background.")
    
    # Add cluster points
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(
            ptsInCurrCluster[:, 0].flatten().A[0],
            ptsInCurrCluster[:, 1].flatten().A[0],
            marker=markerStyle,
            s=90,
            alpha=0.7
        )
    # Add cluster centers (marked with '+')
    ax1.scatter(
        myCentroids[:, 0].flatten().A[0],
        myCentroids[:, 1].flatten().A[0],
        marker='+',
        s=300,
        c='red',
        linewidths=2
    )
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'Club Clustering Result (k={numClust})')
    plt.show()

# Example usage (uncomment to run)
# 1. Prepare address file (e.g., 'addresses.txt') with format: name\tstreet\tcity
# 2. Generate places.txt (note: Yahoo API is deprecated, replace with other APIs first)
# massPlaceFind('addresses.txt')
# 3. Cluster clubs and visualize (requires 'places.txt' and 'Portland.png')
#clusterClubs(numClust=5)
