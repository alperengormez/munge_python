import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def findNearestNeighbor( elementIndex, array):
    """
    elementIndex: the element that we are searching the nearest neighbor of
    array: array
    
    Returns
    -------
    nearestNeighborIndex: the index of the nearestNeighbor
    """
    element = array[elementIndex]
    array = np.delete( array, elementIndex, axis=0) # prevent elementIndex == nearestNeighborIndex
    nearestNeighborIndex = np.argmin( np.linalg.norm( element - array, axis=1))
    
    if nearestNeighborIndex >= elementIndex: # adjust
        nearestNeighborIndex += 1
    return nearestNeighborIndex



def munge( dataset, sizeMultiplier, swapProb, varParam):
    """
    This algorithm was presented in the following paper:
    
    C. Bucilua, R. Caruana, and A. Niculescu-Mizil. Model compression. In Proceedings of the
    12th ACMSIGKDD International Conference on Knowledge Discovery and Data Mining, KDD
    ’06, pages 535–541, New York, NY, USA, 2006. ACM.
    
    Creates a synthetic dataset.
    Continuous attributes should be linearly scaled to [0, 1].
    For now, assumes all attributes are continuous and the dataset is 2D. #TODO
    
    dataset: (numExamples, numAttributes)
    sizeMultiplier: dataset size multiplier
    swapProb: probability of swapping attributes (draw from normal with mean)
    varParam: local variance parameter
    
    Returns
    -------
    synthetic: (sizeMultiplier*numExamples, numAttributes)
    """
    
    numExamples, numAttributes = dataset.shape
    synthetic = np.empty((sizeMultiplier*numExamples, numAttributes))
    
    for i in range( sizeMultiplier):
        tempDataset = np.copy(dataset)
        
        for exampleIndex in range( numExamples):
            nearestNeighborIndex = findNearestNeighbor( exampleIndex, tempDataset)
            
            for j in range( numAttributes):
                if np.random.uniform() < swapProb:
                    example_attr = tempDataset[ exampleIndex, j]
                    closestNeighbor_attr = tempDataset[ nearestNeighborIndex, j]
                    
                    tempDataset[ exampleIndex, j] = np.random.normal( closestNeighbor_attr, abs( example_attr - closestNeighbor_attr) / varParam)
                    tempDataset[ nearestNeighborIndex, j] = np.random.normal( example_attr, abs( example_attr - closestNeighbor_attr) / varParam)
        
        synthetic[i*numExamples:(i+1)*numExamples, :] = tempDataset
    return synthetic




X, y = datasets.load_wine(return_X_y=True)
# we will work on these features
flavanoids = X[:, 6]
proline = X[:, 12]

# linearly scale to [0, 1] (for munge)
flavanoids = (flavanoids - np.min(flavanoids)) / (np.max(flavanoids) - np.min(flavanoids))
proline = (proline - np.min(proline)) / (np.max(proline) - np.min(proline))

plt.scatter( flavanoids, proline, c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Flavanoids')
plt.ylabel('Proline')
plt.title('Original Data')
plt.show()



features = np.stack( (flavanoids, proline), axis=1)
sizeMultiplier = 3
swapProb = 0.75
varParam = 2
synthetic = munge( features, sizeMultiplier, swapProb, varParam)

plt.scatter( synthetic[:, 0], synthetic[:, 1])
plt.xlabel('Flavanoids')
plt.ylabel('Proline')
plt.title('Synthetic Data (sizeMultiplier = 3, swapProb = 0.75, varParam = 2)')
plt.show()
