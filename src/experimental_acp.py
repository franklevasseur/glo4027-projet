import matplotlib
import matplotlib.pyplot as plt

from src.constants import *

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


if __name__ == "__main__":
    bikeDt = pd.read_csv(TRAIN_FILE_PATH)

    print('Colonnes\n', bikeDt.columns)
    print('Sommaire\n', bikeDt.describe())

    bikeDatetime = pd.to_datetime(bikeDt.datetime, format='%Y-%m-%d %H:%M:%S')
    bikeFeatures = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                    'humidity', 'windspeed']
    bikeCounts = ['casual', 'registered', 'count']

    X = bikeDt.loc[:, bikeFeatures].values
    y = bikeDt.loc[:, bikeCounts].values

    print(X)
    print(y)

    # Normalize prior to PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Normalize for PCA.

    pca = PCA(n_components=2)
    bikePCA = pca.fit_transform(X)
    bikePCA = pd.DataFrame(data=bikePCA, columns=['pc 1', 'pc 2'])

    bikeVisual = pd.concat([bikePCA, bikeDt['count']], axis=1)

    print(bikeVisual)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Composante 1', fontsize=15)
    ax.set_ylabel('Composante 2', fontsize=15)
    ax.set_title('PCA', fontsize=20)

    cmap = matplotlib.cm.get_cmap('viridis')
    counts = bikeVisual['count'].values
    normalize = matplotlib.colors.Normalize(vmin=min(counts), vmax=max(counts))
    colors = [cmap(normalize(count)) for count in counts]

    ax.scatter(bikeVisual['pc 1'].values, bikeVisual['pc 2'].values, color=colors)
    plt.show()
