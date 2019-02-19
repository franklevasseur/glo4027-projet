import matplotlib
import matplotlib.pyplot as plt

from src.constants import *

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def plot_pca(bikeDt, bikeVisual, param):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Composante 1', fontsize=15)
    ax.set_ylabel('Composante 2', fontsize=15)

    if param == 'count':
        cmap = matplotlib.cm.get_cmap('viridis')
        counts = bikeVisual['count'].values
        normalize = matplotlib.colors.Normalize(vmin=min(counts), vmax=max(counts))
        colors = [cmap(normalize(count)) for count in counts]

        ax.scatter(bikeVisual['pc 1'].values, bikeVisual['pc 2'].values,
                   color=colors)
        plt.show()

    ax.scatter(bikeVisual['pc 1'].values, bikeVisual['pc 2'].values,
               c=bikeDt['season'].values)
    plt.show()


if __name__ == "__main__":
    bikeDt = pd.read_csv(TRAIN_FILE_PATH)

    print('Colonnes\n', bikeDt.columns)
    print('Sommaire\n', bikeDt.describe())
    print('NaN?', bikeDt.isnull().values.any())

    bikeDatetime = pd.to_datetime(bikeDt.datetime, format='%Y-%m-%d %H:%M:%S')
    print('Duplicate dates check :', bikeDatetime.drop_duplicates().shape == bikeDatetime.shape)

    bikeDt['day'] = bikeDatetime.apply(lambda datetime: datetime.day)
    bikeDt['month'] = bikeDatetime.apply(lambda datetime: datetime.month)
    bikeDt['year'] = bikeDatetime.apply(lambda datetime: datetime.year)
    bikeDt['hour'] = bikeDatetime.apply(lambda datetime: datetime.hour)

    bikeFeatures = ['day', 'month', 'year', 'hour',
                    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                    'humidity', 'windspeed']
    bikeCounts = ['casual', 'registered', 'count']

    # print('Sommaire\n', bikeDt.describe().to_latex())

    bikeFreq = pd.crosstab(index=bikeDt['month'],
                           columns=bikeDt['day'])
    print(bikeFreq)

    # Étude du max
    print(bikeDt.loc[(bikeDt['month'] == 9) &
                     (bikeDt['year'] == 2012) &
                     (bikeDt['hour'] == 18)].to_string())

    # Étude du min
    print(bikeDt.loc[(bikeDt['count'] == 1)].to_string())

    # Humidité (une journée fausse données!)
    print(bikeDt.loc[(bikeDt['humidity'] == 0)].to_string())

    # --- PCA ---
    X = bikeDt.loc[:, bikeFeatures].values
    y = bikeDt.loc[:, bikeCounts].values
    # print(X)
    # print(y)

    # Normalize prior to PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Normalize for PCA.
    print(X.shape)

    pca = PCA(n_components=2)
    # pca = PCA(0.95)
    bikePCA = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)

    bikePCA = pd.DataFrame(data=bikePCA, columns=['pc 1', 'pc 2'])

    bikeVisual = pd.concat([bikePCA, bikeDt['count']], axis=1)

    print(bikeVisual)
    plot_pca(bikeDt, bikeVisual, 'count')
