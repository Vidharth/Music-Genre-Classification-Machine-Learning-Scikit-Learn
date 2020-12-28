# Functions

import numpy as np
import math
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn import linear_model, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
import matplotlib.pyplot as plt


def get_data():
	"""Reads the data from defined filenames features.csv and tracks.csv and returns them
	
	This function gives the dataframes new indices, because some indices are skipped and
	that gives us some troubles when using k-fold cross validation.
	We tested whether the order of original track indices (2.3.5,10,20... etc) is equal in
	both dataframes, and it turns out it is the case. The indices are thus safely removed.
	"""

	features_file = "fma_metadata/features.csv"
	genres_file = "fma_metadata/tracks.csv"

	features = pd.read_csv(features_file, header=[0,1,2], skiprows=[3])
	print(f"The shape of features is: {features.shape}")

	genres = pd.read_csv(genres_file, header=[0,1], skiprows=[2])
	# Picked the most generic "genres" column
	genres = genres["track", "genres"]
	print(f"The shape of genres is: {genres.shape}")

	# The number of rows should be equal in the shapes (106574)
	return features, genres

def drop_rows(features, genres, drop_indices):
	"""This function drops rows from features and genres."""

	genres = genres.drop(labels=drop_indices, axis=0)
	features = features.drop(labels=drop_indices, axis=0) # We have to remove the same track from the features dataframe

	# Reset indices, so that we do not get NaNs in shufflesplit
	genres = genres.reset_index(drop=True).squeeze()
	features = features.set_index(pd.Index([i for i in range(len(features))]), drop=True)
	return features, genres

def clean_data(features, genres):
	"""Some rows in the csv files have missing values or worse, wrong values. For example,
	see tracks 1020, 1021, 1022 and 1023 in tracks.csv. Their genres are messewd us, because
	all values seem to have shifted one column to the right in these rows.
	Therefore, we check whether the genre in the row has the correct format:
	[x] where x is an integer or multiple integers seperated by commas.
	""" 
	
	features = features.drop(labels=('feature', 'statistics', 'number'), axis=1) # Drop track_id

	drop_indices = set()
	for index, genre in genres.items():
		if not (genre.startswith("[") and genre.endswith("]") and len(genre) > 2):
			# This is a row with a missing or corrupted value, we remove the row
			drop_indices.add(index)

	features, genres = drop_rows(features, genres, drop_indices)

	print(f"The shape of features after cleaning is: {features.shape}")
	print(f"The shape of genres after cleaning is: {genres.shape}")

	return features, genres

def shufflesplit_data(features, genres, test_ratio=0.2):
	"""Shuffles the data first, according to a permutation.
	This guarantees that both the features data and the genres will be
	shuffled the same way.

	20% of the data will be used as a final testing set.
	This way, we can tell how well the algorithm generalizes.
	"""

	np.random.seed(42) # For reproduction purposes
	permutation = np.random.permutation(len(features))
	features = features.reindex(permutation)
	genres = genres.reindex(permutation)

	# Lengths of features and genres are the same
	seen_index = [i for i in range(math.floor((1.0-test_ratio) * len(features)))]
	unseen_index = [i for i in range(math.floor((1.0-test_ratio) * len(features)), len(features))]

	seen_features, unseen_features = features.reindex(seen_index), features.reindex(unseen_index)
	seen_genres, unseen_genres = genres.reindex(seen_index), genres.reindex(unseen_index)

	return seen_features, unseen_features, seen_genres, unseen_genres

def new_single_genres(genres, val):
    """Takes the genres list and returns only one genre back if multiple genres are present
    Also has the parameter val with values "high" and "low"
    High picks the genres belonging to the existing genres with the highest examples count
    Low picks the genres belonging to the existing genres with the least examples count"""
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
    return genres.astype('int')

def new_single_genres(genres, val):
    """Takes the genres list and returns only one genre back if multiple genres are present
    Also has the parameter val with values "high" and "low"
    High picks the genres belonging to the existing genres with the highest examples count
    Low picks the genres belonging to the existing genres with the least examples count"""
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

    genres = genres.astype('int')

    return genres

def remove_infrequent_classes(features, genres, threshold):
	"""This function removes instances from classes with num_instances
	below a threshold.
	"""

	drop_indices = set()
	for genre_code in range(1, max(genres)+1):
		count = genres[genres == genre_code].count()

		if count < threshold:
			# Remove instances from genres with fewer instances than the threshold
			drop_indices.update(set(genres[genres == genre_code].index))

	features, genres = drop_rows(features, genres, drop_indices)
	print(f"The shape of features after removing infrequent classes is: {features.shape}")
	print(f"The shape of genres after removing infrequent classes is: {genres.shape}")
	return features, genres

def normalize_features(features):
	"""This function normalizes all the values of the features.
	"""
	cols = features.columns

	normalizer = preprocessing.Normalizer()
	np_scaled = normalizer.fit_transform(features)
	features_normalized = pd.DataFrame(np_scaled, columns = cols)

	return features_normalized

def select_features(features, feature_combination):
	print(f"Selecting feature combination: {feature_combination}")
	features = features[feature_combination] #mfcc seems to be the most important
	print(f"The shape of features after selecting features is: {features.shape}")
	return features

def define_feature_combinations():
	"""This function defines feature combinations. These feature combinations
	were tested by using a linear regression classifier to compute the
	cross-validation scores (the function find_best_feature_combination()).
	 The feature combination with the highest
	cross-validation score was kept and used in the pipeline.

	The best feature combination using this method was found to be: 
	['chroma_cens', 'mfcc', 'spectral_contrast']
	This function became obsolete after that point, but it is here for completeness.
	"""

	# The features itself go in first
	feature_combinations = [['chroma_stft'], ['chroma_cqt'], ['chroma_cens'],
							['mfcc'], ['rmse'],	['spectral_centroid'],
							['spectral_bandwidth'], ['spectral_contrast'],
							['spectral_rolloff'], ['tonnetz'], ['zcr']]

	# Then, new feature combinations are added to that list (combis of 2)
	new_feature_combinations = []
	for i, f in enumerate(feature_combinations[:-1]):
		for j in range(i+1, len(feature_combinations)):
			f2 = feature_combinations[j]
			new_fc = f.copy()
			new_fc.extend(f2)
			new_feature_combinations.append(new_fc)

	# Then, even more new feature combinations are added (combis of 3)
	for i, f in enumerate(feature_combinations[:-2]):
		for j in range(i+1, len(feature_combinations) - 1):
			for k in range(j+1, len(feature_combinations)):
				f2 = feature_combinations[j]
				f3 = feature_combinations[k]
				new_fc = f.copy()
				new_fc.extend(f2)
				new_fc.extend(f3)
				new_feature_combinations.append(new_fc)

	feature_combinations.extend(new_feature_combinations)
	return feature_combinations

def find_best_feature_combination(features, genres):
	"""This function was used to find out which combination of features
	yielded the best results. See define_feature_combinations() for more details.
	This function became obsolete after finding out 
	['chroma_cens', 'mfcc', 'spectral_contrast'] was the combination that yielded
	the highest mean cv-score, but it remains here for completeness.
	"""

	feature_combinations = define_feature_combinations()

	# Keep track of cv scores of classifiers with different feature combinations.
	mean_cross_val_scores = [] 
	for feature_combi in feature_combinations:
		selected_features = select_features(features, feature_combi)

		# Split data into kfold training/validation and final test set
		features_train, features_test, genres_train, genres_test = shufflesplit_data(selected_features, genres, test_ratio=0.2)
		
		# An alpha value of 0 makes ridge regression equivalent to standard linear regression
		clf = linear_model.RidgeClassifier(alpha=0)
		scores = cross_val_score(clf, features_train, genres_train, cv=10)
		print(f"Cross val score was {scores.mean():.4f} (+/- {scores.std()*2:.4f}).")
		mean_cross_val_scores.append(scores.mean())
		clf.fit(features_train, genres_train)
		test_acc = clf.score(features_test, genres_test) # Test acc should be calculated after cross-validation
		print(f"Test accuracy for the RidgeClassifier with alpha {alpha} was: {test_acc:.4f}")

	print(f"The mean cv scores: {mean_cross_val_scores}")
	mean_cross_val_scores = np.array(mean_cross_val_scores)
	print(f"The index of the maximum value of the mean cross-validation scores is: {np.argmax(mean_cross_val_scores)},\
	 which means the best combination of features is: {feature_combinations[np.argmax(mean_cross_val_scores)]}.")