from sklearn.decomposition import PCA
import numpy as np

def varianceStudio(data, pca_threshold):
    pca = PCA()
    pca.fit(data)
    # variances = pca.explained_variance_
    # print("Mean: {} (+/- {})".format(np.mean(variances), np.std(variances)))
    # condition = variances > pca_threshold
    # n_components = len(np.extract(condition, variances))
    print(pca.n_components_)


    return pca.n_components_

def componentSelection(data, n_components):
    pca = PCA(n_components= n_components)
    new_data = pca.fit_transform(data)

    return new_data