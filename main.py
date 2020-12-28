"""All functions that are called here can be found in pipeline.py.
This file contains the main algorithm used.
This file follows the pseudo-code mentioned in the report (Algorithm 1)."""

from pipeline import *
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score

# 1) Get the data
features, genres = get_data()

# 2) Clean the data
features, genres = clean_data(features, genres)

# 3) Some songs have multiple genres. Pick one of the following strategies to singlify
genres = new_single_genres(genres, "high") 	# Strategy C 

# 4) Remove infrequent classes from the dataset
features, genres = remove_infrequent_classes(features, genres, threshold=2500)
print(f"The set of genres that is left is: {genres.unique()}")

# 5) Select the best combination of features
# see define_feature_combinations() and find_best_feature_combination() in pipeline.py
features = select_features(features, ['chroma_cens', 'mfcc', 'spectral_contrast'])

# 6) Shuffle and split the data into a train and test set
# The test set will not be used during the cross-validation
features_train, features_test, genres_train, genres_test = shufflesplit_data(features, genres, test_ratio=0.2)

## Ridge Regression Classification
for alpha in [0, 1, 2, 3, 4]:
	clf = RidgeClassifier(alpha=alpha)
	scores = cross_val_score(clf, features_train, genres_train, cv=10)
	print(f"Cross val score was {scores.mean():.4f} (+/- {scores.std()*2:.4f}).")

	# 7) Perform training of a classifier
	clf.fit(features_train, genres_train)

	# 8) Evaluate the classifier
	test_acc = clf.score(features_test, genres_test) # Test acc should be calculated after cross-validation
	print(f"Test accuracy for the RidgeClassifier with alpha {alpha} was: {test_acc:.4f}")

## Random Forest Classification
best_acc = 0
ccp_alpha = 0
for max_depth in range(14,30,2):
	for min_impurity_decrease in np.arange(0, 1*10**(-7), 2*10**(-8)):
		print("New Forest")
		clf = RandomForestClassifier(n_estimators = 100, criterion = "entropy", n_jobs = 4, random_state = 0,\
			ccp_alpha = ccp_alpha, max_depth = max_depth, min_impurity_decrease = min_impurity_decrease)
		scores = cross_val_score(clf, features_train, genres_train, cv=10)
		avg_score = scores.mean()
		print(f"Average test accuracy for the RandomForestClassifier was: {avg_score:.4f}")
		if avg_score > best_acc:
			best_acc = avg_score
			ccp_alpha_opt = ccp_alpha
			max_depth_opt = max_depth
			min_impurity_decrease_opt = min_impurity_decrease
				
clf = RandomForestClassifier(n_estimators = 100, criterion = "entropy", n_jobs = 4, random_state = 0,\
				ccp_alpha = ccp_alpha_opt, max_depth = max_depth_opt, min_impurity_decrease = min_impurity_decrease_opt)
clf.fit(features_train,genres_train)
test_acc = clf.score(features_test.values, genres_test.values)
print(f"Optimal solution for the RandomForestClassifier with average_test_acc: {test_acc:.4f} was found with ccp_alpha: {ccp_alpha_opt:.2f}\
	, max depth of: {max_depth_opt} and min_impurity_decrease of {min_impurity_decrease_opt:.8f}")