"""
ML Project
kNN Classifier
"""

import pandas as pd
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def get_data():
    """Reads the data from defined file names features.csv and tracks.csv and returns them

    This function gives the data frames new indices, because some indices are skipped and
    that gives us some troubles when using k-fold cross validation.
    We tested whether the order of original track indices (2.3.5,10,20... etc) is equal in
    both data frames, and it turns out it is the case. The indices are thus safely removed.
    """
    features_file = "fma_metadata/features.csv"
    genres_file = "fma_metadata/tracks.csv"

    features = pd.read_csv(features_file, header=[0, 1, 2], skiprows=[3])
    print("The shape of features is:{}".format(features.shape))

    genres = pd.read_csv(genres_file, header=[0, 1], skiprows=[2])
    # Picked the most generic "genres" column
    genres = genres["track", "genres"]
    print("The shape of genres is:{}".format(genres.shape))

    # The number of rows should be equal in the shapes (106574)
    return features, genres


def clean_data(features, genres):
    """Some rows in the csv files have missing values or worse, wrong values. For example,
    see tracks 1020, 1021, 1022 and 1023 in tracks.csv. Their genres are messed up, because
    all values seem to have shifted one column to the right in these rows.
    Therefore, we check whether the genre in the row has the correct format:
    [x] where x is an integer or multiple integers separated by commas."""

    features = features.drop(labels=('feature', 'statistics', 'number'), axis=1)  # Drop track_id

    drop_indices = []
    for index, genre in genres.items():
        if not (genre.startswith("[") and genre.endswith("]") and len(genre) > 2):
            # This is a row with a missing or corrupted value, we remove the row
            drop_indices.append(index)

    genres = genres.drop(labels=drop_indices, axis=0)
    features = features.drop(labels=drop_indices,
                             axis=0)  # We have to remove the same track from the features data frame

    # Reset indices, so that we do not get NaNs in shuffle split
    genres = genres.reset_index(drop=True).squeeze()
    features = features.set_index(pd.Index([i for i in range(len(features))]), drop=True)

    print("The shape of features after cleaning is:{}".format(features.shape))
    print("The shape of genres after cleaning is:{}".format(genres.shape))

    return features, genres


def single_genres(genres):
    """Takes the genres list and returns only one genre back if multiple genres are present"""
    for index, genre in genres.items():
        split = genre.split(",")
        new_genre = split[0]
        new_genre = new_genre.strip("[]")
        genres[index] = int(new_genre)
    print("The shape of genres after single is:{}".format(genres.shape))

    return genres


def new_single_genres(genres, val):
    """Takes the genres list and returns only one genre back if multiple genres are present"""
    genres_file = "fma_metadata/genres.csv"
    reference_genres = pd.read_csv(genres_file)
    reference_tracks = reference_genres.iloc[:, 1]
    reference_genres = reference_genres.iloc[:, 0]

    for index, genre in genres.items():
        split = genre.split(",")
        if len(split) == 1:
            new_genre = split[0]
            new_genre = new_genre.strip("[]")
            genres[index] = int(new_genre)
        elif len(split) > 1:
            new_genre = [int(item.strip(" [ ] ")) for item in split]
            count = {}
            for indices, value in reference_genres.items():
                if value in new_genre:
                    count[value] = reference_tracks[indices]
            counts = {k: v for k, v in sorted(count.items(), key=lambda item: item[1])}
            if val == "high":
                genres[index] = int(list(counts.keys())[-1])
            elif val == "low":
                genres[index] = int(list(counts.keys())[0])
    print("The shape of genres after single is:{}".format(genres.shape))

    return genres


def threshold_genres_classes(features, genres, threshold):
    """Limits the number of classes as per the threshold"""
    drop_indices = []
    drop_index = []
    value_counts = genres.value_counts()
    for index, value in value_counts.items():
        if value < threshold:
            drop_index.append(index)
    for values in drop_index:
        for index, value in genres.items():
            if value == values:
                drop_indices.append(index)

    genres = genres.drop(labels=drop_indices, axis=0)
    features = features.drop(labels=drop_indices,
                             axis=0)  # We have to remove the same track from the features data frame

    # Reset indices, so that we do not get NaNs in shuffle split
    genres = genres.reset_index(drop=True).squeeze()
    features = features.set_index(pd.Index([i for i in range(len(features))]), drop=True)

    print("The shape of features after threshold is:{}".format(features.shape))
    print("The shape of genres after threshold is:{}".format(genres.shape))

    return features, genres


def accuracy(c_m):
    """Accuracy of Confusion Matrix"""
    diagonal_sum = c_m.trace()
    sum_of_all_elements = c_m.sum()

    return diagonal_sum / sum_of_all_elements


Features, Genres = get_data()
Features, Genres = clean_data(Features, Genres)
Genres = new_single_genres(Genres, "high")
Features, Genres = threshold_genres_classes(Features, Genres, 2500)
Genres = Genres.astype('int')

print("Genre Classes", len(set(Genres)))

"""
Features = normalize(Features)
pca = PCA(n_components=100)
pca.fit(Features)
Features = pca.transform(Features)
"""

Scale = StandardScaler()
Scale.fit(Features)
Features = Scale.transform(Features)
Encoder = LabelEncoder()
Encoder.fit(Genres)
Genres = Encoder.transform(Genres)

SeenFeatures, UnseenFeatures, SeenGenres, UnseenGenres = train_test_split(Features, Genres, train_size=0.8,
                                                                          stratify=Genres)

print(SeenFeatures)
print(SeenFeatures.shape)
print(SeenGenres)
print(SeenGenres.shape)
print("Genre Classes", len(set(SeenGenres)))
print(UnseenFeatures)
print(UnseenFeatures.shape)
print(UnseenGenres)
print(UnseenGenres.shape)
print("Genre Classes", len(set(UnseenGenres)))

KNNClassifier = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
print(KNNClassifier)
KNNClassifier.fit(SeenFeatures, SeenGenres)
PredictGenres = KNNClassifier.predict(UnseenFeatures)
cm = confusion_matrix(PredictGenres, UnseenGenres)

print(KNNClassifier.score(SeenFeatures, SeenGenres))
print(KNNClassifier.score(UnseenFeatures, UnseenGenres))
print("Accuracy:", accuracy(cm))
print(classification_report(PredictGenres, UnseenGenres))
