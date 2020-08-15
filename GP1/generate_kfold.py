# -*- coding: utf-8 -*-
"""Uses Synthetic Regression to discover new methods of calculating Lattice Thermal Conductivity (LTC)

Written by loftis
"""
import numpy as np
import pandas as pd
from fastsr.estimators.symbolic_regression import SymbolicRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os

FOLDER_NAME = "top_models_kfold_cv"

try:
    if not os.path.isdir(FOLDER_NAME):
		os.mkdir(FOLDER_NAME)
except FileExistsError:
    pass


print("Running...")

def calc_score(sr, X_test, Y_test, test_index=None, final_check=False):
	predictions = None
	if isinstance(X_test, np.ndarray):
		predictions = sr.predict(X_test)
	else:
		predictions = sr.predict(X_test.values)
	slack_values = []
	if test_index is not None:
		slack_values = dataset.iloc[test_index]["κ"].astype("float64").values
	else:
		slack_values = dataset.loc[X_test.axes[0].tolist(), "κ"].values

	slack_mae = mean_absolute_error(Y_test, slack_values)
	try:
		model_mae = mean_absolute_error(Y_test, predictions)
	except ValueError:
		print("ERROR: Encountered NaN, Infinity, or values larger than float64 supports")
		return None, None
	else:
		if not final_check:
			print("Slack:", slack_mae)
			print("Model:", model_mae)

		if model_mae < slack_mae:
			print("Currently outperforming slack model by", (slack_mae - model_mae))
		else:
			print("Currently underperforming by", (model_mae - slack_mae))
		return model_mae, slack_mae


dataset = pd.read_csv("../dataset.csv", delimiter=",")

#Sort the dataset by the thermal conductivity values
dataset = dataset.sort_values('κref.')

X = dataset.iloc[:, 2:23]
Y = dataset["κref."].astype("float64")

slack_values = dataset.loc[:, "κ"]

X_train = pd.DataFrame().reindex_like(X)
X_test = pd.DataFrame().reindex_like(X)
Y_train = pd.Series(name="κref.")
Y_test = pd.Series(name="κref.")

X_train = X_train.iloc[0:0]
X_test = X_test.iloc[0:0]

slack_train_values = pd.Series()
slack_test_values = pd.Series()

sr = SymbolicRegression(ngen=500, pop_size=2000, max_height=7, mutation_probability=0.3, crossover_probability=0.7)

#Kfolds cross validation, with 10 splits
kf = KFold(n_splits=10, shuffle=True)

scores = []
slack_scores = []

total_sum = 0

# Loop through the KFolds splits, and train/test the model over them
for train_index, test_index in kf.split(X):
	X_temp_train, X_temp_test = X.iloc[train_index].values, X.iloc[test_index].values
	Y_temp_train, Y_temp_test = Y.iloc[train_index].values, Y.iloc[test_index].values
	print(X_temp_train.shape[0])
	total_sum += X_temp_train.shape[0]
	X_train = X_train.append(pd.DataFrame(X_temp_train, columns=X.columns))
	X_test = X_test.append(pd.DataFrame(X_temp_test, columns=X.columns))
	Y_train = Y_train.append(pd.Series(Y_temp_train))
	Y_test = Y_test.append(pd.Series(Y_temp_test))

	slack_train_values = slack_train_values.append(pd.Series(dataset.loc[train_index, "κ"].values))
	slack_test_values = slack_test_values.append(pd.Series(dataset.loc[test_index, "κ"].values))
	sr.fit(X_temp_train, Y_temp_train)
	score, slack_score = calc_score(sr, X_temp_test, Y_temp_test, test_index)
	scores.append(score)
	slack_scores.append(slack_score)
	print('Score: {}'.format(score))
	print(str(sr.best_individuals_[0]))

scores_df = pd.DataFrame(data=scores, columns=["K-Fold_Scores"])

print("=================================")

overall_score = sum(scores) / len(scores)
overall_slack_score = sum(slack_scores) / len(slack_scores)
print("Average Score from K-Folds:", overall_score)
print("Slack Score from K-Folds:", overall_slack_score)
print("")

test_score, slack_mae = calc_score(sr, X_test, Y_test, final_check=True)
print("Testing set score:", test_score)
print("Slack model:", slack_mae)

train_score, slack_train_score = calc_score(sr, X_train, Y_train, final_check=True)

training_data = pd.concat([X_train, slack_train_values.rename("κ"), Y_train.rename("κref.")], axis=1)
test_data = pd.concat([X_test, slack_test_values.rename("κ"), Y_test.rename("κref.")], axis=1)

print(training_data.columns)

# if test_score <= slack_mae and train_score <= slack_train_score:
# Save the model
num = 0
folder = FOLDER_NAME + "/" + str(num)

while os.path.isdir(folder):
	#If the folder with that ID exists, increment by 1, and use that ID instead
	folder = FOLDER_NAME + "/" + str(num + 1)
	num += 1

os.mkdir(folder)
sr.save(folder + "/model")
training_data.to_csv(folder + "/training_data.csv")
test_data.to_csv(folder + "/test_data.csv")
scores_df.to_csv(folder + "/learning_scores.csv", index=False)
