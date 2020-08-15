# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os
import pickle
import math

FOLDER_NAME = "top_models_hybrid_cv"

try:
	if not os.path.isdir(FOLDER_NAME):
		os.mkdir(FOLDER_NAME)
except FileExistsError:
	pass

def save_model(rfr, filepath):
	with open(filepath, 'wb') as file:
		pickle.dump(rfr, file)

if "-h" in sys.argv or "--help" in sys.argv:
	print("LTC Symbolic Regression Help Menu")
	print("     --bottom-testing: Using the bottom 20% of the dataset as testing set")
	print("     --top-testing: Using the top 20% of the dataset as testing set")
	print("     --top-bottom-testing: Using the top 10% and bottom 10% of the dataset as testing set")
	print("     --middle-testing: Using the middle 20% of the dataset as testing set")

BOTTOM_TESTING_ARGSTRING = "--bottom-testing"
TOP_TESTING_ARGSTRING = "--top-testing"
TOP_BOTTOM_TESTING_ARGSTRING = "--top-bottom-testing"
MIDDLE_TESTING_ARGSTRING = "--middle-testing"

bottom_eval = False
top_eval = False
top_bottom_eval = False
true_middle_eval = False

if BOTTOM_TESTING_ARGSTRING in sys.argv and not (TOP_TESTING_ARGSTRING in sys.argv or TOP_BOTTOM_TESTING_ARGSTRING in sys.argv or MIDDLE_TESTING_ARGSTRING in sys.argv):
	bottom_eval = True
if TOP_TESTING_ARGSTRING in sys.argv and not (BOTTOM_TESTING_ARGSTRING in sys.argv or TOP_BOTTOM_TESTING_ARGSTRING in sys.argv or MIDDLE_TESTING_ARGSTRING in sys.argv):
	top_eval = True
if TOP_BOTTOM_TESTING_ARGSTRING in sys.argv and not (BOTTOM_TESTING_ARGSTRING in sys.argv or TOP_TESTING_ARGSTRING in sys.argv or MIDDLE_TESTING_ARGSTRING in sys.argv):
	top_bottom_eval = True
if MIDDLE_TESTING_ARGSTRING in sys.argv and not (BOTTOM_TESTING_ARGSTRING in sys.argv or TOP_TESTING_ARGSTRING in sys.argv or TOP_BOTTOM_TESTING_ARGSTRING in sys.argv):
	true_middle_eval = True

#If the user didn't specify a testing set
if (not bottom_eval) and (not top_eval) and (not top_bottom_eval) and (not true_middle_eval):
	print("Please specify a testing set. If you are not sure how to specify a testing set, run \"python " + __file__ + " -h\"" )
	sys.exit()

print("Running...")

def calc_score(rfr, X_test, Y_test, test_index=None, final_check=False):
	predictions = None
	if isinstance(X_test, np.ndarray):
		predictions = rfr.predict(X_test)
	else:
		predictions = rfr.predict(X_test.values)
	slack_values = []
	if test_index is not None:
		slack_values = dataset.iloc[test_index]["κ"].astype("float64").values
	else:
		slack_values = dataset.loc[X_test.axes[0].tolist(), "κ"].values

	slack_rmse = math.sqrt(mean_squared_error(Y_test, slack_values))
	try:
		model_rmse = math.sqrt(mean_squared_error(Y_test, predictions))
	except ValueError:
		print("ERROR: Encountered NaN, Infinity, or values larger than float64 supports")
		return None, None
	else:
		# slack_rmse = mean_squared_error(Y_test,slack_values)
		# model_rmse = mean_squared_error(Y_test,slack_values)

		if not final_check:
			print("Slack:", slack_rmse)
			print("Model:", model_rmse)

		if model_rmse < slack_rmse:
			print("Currently outperforming slack model by", (slack_rmse - model_rmse))
		else:
			print("Currently underperforming by", (model_rmse - slack_rmse))
		return model_rmse, slack_rmse

def preprocess_data_holdout_random(dataset):
	X = dataset.iloc[:, 1:23]
	Y = dataset["κref."].astype("float64")

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, shuffle=True)

	return X_train, X_test, Y_train, Y_test

def preprocess_data_holdout_top(dataset):
	X = dataset.iloc[:, 1:23]
	Y = dataset["κref."].astype("float64")

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=False)

	return X_train, X_test, Y_train, Y_test

def preprocess_data_holdout_middle(dataset):
	X = dataset.iloc[:, 1:23].values
	Y = dataset["κref."].astype("float64")

	first_ten_percent = dataset.iloc[:int(0.10 * dataset.shape[0])]
	last_ten_percent = dataset.iloc[int(0.90 * dataset.shape[0]):]

	testing = pd.concat([first_ten_percent, last_ten_percent], axis=0)
	training = dataset.iloc[int(0.10 * dataset.shape[0]):int(0.90 * dataset.shape[0])]

	X_train = training.iloc[:, 1:23]
	Y_train = training["κref."].astype("float64")

	X_test = testing.iloc[:, 1:23]
	Y_test = testing["κref."].astype("float64")

	return X_train, X_test, Y_train, Y_test

def preprocess_data_holdout_true_middle(dataset):
	X = dataset.iloc[:, 1:23].values
	Y = dataset["κref."].astype("float64")

	first_fourty_percent = dataset.iloc[:int(0.40 * dataset.shape[0])]
	last_fourty_percent = dataset.iloc[int(0.60 * dataset.shape[0]):]

	training = pd.concat([first_fourty_percent, last_fourty_percent], axis=0)
	testing = dataset.iloc[int(0.40 * dataset.shape[0]):int(0.60 * dataset.shape[0])]

	X_train = training.iloc[:, 1:23]
	Y_train = training["κref."].astype("float64")

	X_test = testing.iloc[:, 1:23]
	Y_test = testing["κref."].astype("float64")

	return X_train, X_test, Y_train, Y_test

def preprocess_data_holdout_bottom(dataset):
	X = dataset.iloc[:, 2:23]
	Y = dataset["κref."].astype("float64")

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80, shuffle=False)

	return X_test, X_train, Y_test, Y_train

def preprocess_space_groups(dataset):
	space_groups = dataset.loc[:, "space group"]
	labels, uniques = pd.factorize(space_groups)
	dataset["space group"] = labels
	return dataset.drop("Unnamed: 1", axis=1)


dataset = pd.read_csv("../dataset.csv", delimiter=",")
#Sort the dataset by the thermal conductivity values
dataset = dataset.sort_values('κref.')
dataset = preprocess_space_groups(dataset)

X_train, X_test, Y_train, Y_test = (None, None, None, None)

if bottom_eval:
	X_train, X_test, Y_train, Y_test = preprocess_data_holdout_bottom(dataset)
elif top_eval:
	X_train, X_test, Y_train, Y_test = preprocess_data_holdout_top(dataset)
elif top_bottom_eval:
	X_train, X_test, Y_train, Y_test = preprocess_data_holdout_middle(dataset)
elif true_middle_eval:
	X_train, X_test, Y_train, Y_test = preprocess_data_holdout_true_middle(dataset)
else:
	print("Unable to find specified testing set. Terminating early.")
	sys.exit()

slack_values = dataset.loc[:, "κ"]

slack_test_values = dataset.loc[X_test.axes[0].tolist(), "κ"]
slack_train_values = dataset.loc[~dataset["κ"].isin(slack_test_values), "κ"]

holdout_data = pd.concat([X_test, slack_test_values, Y_test], axis=1)
training_data = pd.concat([X_train, slack_train_values, Y_train], axis=1)

rfr = RandomForestRegressor(n_estimators=100, criterion="mae")

#Kfolds cross validation, with 5 splits
kf = KFold(n_splits=5)

scores = []
scores_on_holdout_set = []
slack_scores = []

# Loop through the KFolds splits, and train/test the model over them
for train_index, test_index in kf.split(X_train):
	X_temp_train, X_temp_test = X_train.values[train_index], X_train.values[test_index]
	Y_temp_train, Y_temp_test = Y_train.values[train_index], Y_train.values[test_index]
	rfr.fit(X_temp_train, Y_temp_train)
	score, slack_score = calc_score(rfr, X_temp_test, Y_temp_test, test_index)
	scores_on_holdout_set.append(mean_absolute_error(Y_test, rfr.predict(X_test.values)))
	scores.append(score)
	slack_scores.append(slack_score)
	print('Score: {}'.format(score))

scores_df = pd.DataFrame(data=np.array([scores, scores_on_holdout_set]).transpose(), columns=["K-Fold_Scores", "Holdout_Scores"])

print("=================================")

overall_score = sum(scores) / len(scores)
overall_slack_score = sum(slack_scores) / len(slack_scores)
print("Average Score from K-Folds:", overall_score)
print("Slack Score from K-Folds:", overall_slack_score)
print("")

test_score, slack_rmse = calc_score(rfr, X_test, Y_test, final_check=True)
print("Holdout test score:", test_score)
print("Slack model:", slack_rmse)
names = "V,M,Θe,γe,n,np,A,T".split(",")

train_score, slack_train_score = calc_score(rfr, X_train, Y_train, final_check=True)

# if test_score <= slack_rmse and train_score < slack_train_score:
#Save the model
num = 0
folder = FOLDER_NAME + "/" + str(num)

while os.path.isdir(folder):
	#If the folder with that ID exists, increment by 1, and use that ID instead
	num += 1
	folder = FOLDER_NAME + "/" + str(num)

os.mkdir(folder)
save_model(rfr, folder + "/model")
holdout_data.to_csv(folder + "/holdout_data.csv")
training_data.to_csv(folder + "/training_data.csv")
scores_df.to_csv(folder + "/learning_scores.csv", index=False)
