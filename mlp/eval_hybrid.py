import sys

BOTTOM_TESTING_ARGSTRING = "--bottom-testing"
TOP_TESTING_ARGSTRING = "--top-testing"
TOP_BOTTOM_TESTING_ARGSTRING = "--top-bottom-testing"
MIDDLE_TESTING_ARGSTRING = "--middle-testing"

interactive = False
flag_missing_vars = False
exclude_underperformers = False
disp_testing = False
bottom_eval = False
top_bottom_eval = False
top_eval = False
true_middle_eval = False
use_rmse = False

model_dir = "top_models_hybrid_cv"

if "-i" in sys.argv or "--interactive" in sys.argv:
	interactive = True
if "-h" in sys.argv or "--help" in sys.argv:
	print("LTC Multilayer-Perceptron Help Menu")
	print("Usage: python eval_hybrid.py [arguments]")
	print("Options/arguments:")
	print("     -h or --help: Displays this menu")
	print("     -i or --interactive: Launch in interactive mode. Allows for plotting and displaying of equations interactively.")
	print("     -e or --exclude: Omits models that did not outperform the Slack Berman model when scored against the entire dataset")
	print("     -t or --testing: Displays testing MAE and R2 insead of average MAE and R2")
	print("     -r or --use-rmse: Displays RMSE instead of MAE")
	print("     --bottom-testing: Using the bottom 20% of the dataset as testing set")
	print("     --top-testing: Using the top 20% of the dataset as testing set")
	print("     --top-bottom-testing: Using the top 10% and bottom 10% of the dataset as testing set")
	print("     --middle-testing: Using the middle 20% of the dataset as testing set")
	sys.exit()

if "-e" in sys.argv or "--exclude" in sys.argv:
	exclude_underperformers = True
if "-t" in sys.argv or "--testing" in sys.argv:
	disp_testing = True
if BOTTOM_TESTING_ARGSTRING in sys.argv and not (TOP_TESTING_ARGSTRING in sys.argv or TOP_BOTTOM_TESTING_ARGSTRING in sys.argv or MIDDLE_TESTING_ARGSTRING in sys.argv):
	bottom_eval = True
if TOP_TESTING_ARGSTRING in sys.argv and not (BOTTOM_TESTING_ARGSTRING in sys.argv or TOP_BOTTOM_TESTING_ARGSTRING in sys.argv or MIDDLE_TESTING_ARGSTRING in sys.argv):
	top_eval = True
if TOP_BOTTOM_TESTING_ARGSTRING in sys.argv and not (BOTTOM_TESTING_ARGSTRING in sys.argv or TOP_TESTING_ARGSTRING in sys.argv or MIDDLE_TESTING_ARGSTRING in sys.argv):
	top_bottom_eval = True
if MIDDLE_TESTING_ARGSTRING in sys.argv and not (BOTTOM_TESTING_ARGSTRING in sys.argv or TOP_TESTING_ARGSTRING in sys.argv or TOP_BOTTOM_TESTING_ARGSTRING in sys.argv):
	true_middle_eval = True
if "-r" in sys.argv or "--use-rmse" in sys.argv:
	use_rmse = True

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import math
import pickle

with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=FutureWarning)
	from keras.models import Sequential
	from keras.layers import Dense, Activation, Dropout
	from keras.utils import np_utils
	import keras.models

def load_model(filepath):
	return keras.models.load_model(filepath)

def preprocess_data_holdout_top(dataset):
	raw_X = dataset.iloc[:, 2:23]

	scaler = MinMaxScaler()
	scaler.fit(raw_X)

	X = pd.DataFrame(scaler.transform(dataset.iloc[:, 2:23]), columns=raw_X.columns)

	space_groups = dataset.loc[:, "space group"]
	dummies = pd.get_dummies(space_groups)
	X = pd.concat([dummies, X], axis=1)
	X.index = raw_X.index

	Y = dataset["κref."].astype("float64")

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=False)

	print(X.columns)

	global test_indices, train_indices
	test_indices = Y_test.index
	train_indices = Y_train.index

	return X_train, X_test, Y_train, Y_test


def preprocess_data_holdout_middle(dataset):
	raw_X = dataset.iloc[:, 2:23]

	scaler = MinMaxScaler()
	scaler.fit(raw_X)

	X = pd.DataFrame(scaler.transform(dataset.iloc[:, 2:23]), columns=raw_X.columns)

	space_groups = dataset.loc[:, "space group"]
	dummies = pd.get_dummies(space_groups)
	X = pd.concat([dummies, X], axis=1)
	X.index = raw_X.index

	Y = dataset["κref."].astype("float64")

	new_ds = pd.concat([X, Y], axis=1)

	first_ten_percent = new_ds.iloc[:int(0.10 * dataset.shape[0])]
	last_ten_percent = new_ds.iloc[int(0.90 * dataset.shape[0]):]

	testing = pd.concat([first_ten_percent, last_ten_percent], axis=0)
	training = new_ds.iloc[int(0.10 * new_ds.shape[0]):int(0.90 * new_ds.shape[0])]


	X_train = training.iloc[:, :-1]
	Y_train = training["κref."].astype("float64")

	X_test = testing.iloc[:, :-1]
	Y_test = testing["κref."].astype("float64")

	global test_indices, train_indices
	test_indices = Y_test.index
	train_indices = Y_train.index

	return X_train, X_test, Y_train, Y_test

def preprocess_data_holdout_true_middle(dataset):
	raw_X = dataset.iloc[:, 2:23]

	scaler = MinMaxScaler()
	scaler.fit(raw_X)

	X = pd.DataFrame(scaler.transform(dataset.iloc[:, 2:23]), columns=raw_X.columns)

	space_groups = dataset.loc[:, "space group"]
	dummies = pd.get_dummies(space_groups)
	X = pd.concat([dummies, X], axis=1)
	X.index = raw_X.index

	Y = dataset["κref."].astype("float64")

	new_ds = pd.concat([X, Y], axis=1)

	first_fourty_percent = dataset.iloc[:int(0.40 * dataset.shape[0])]
	last_fourty_percent = dataset.iloc[int(0.60 * dataset.shape[0]):]

	training = pd.concat([first_fourty_percent, last_fourty_percent], axis=0)
	testing = dataset.iloc[int(0.40 * dataset.shape[0]):int(0.60 * dataset.shape[0])]

	X_train = training.iloc[:, :-1]
	Y_train = training["κref."].astype("float64")

	X_test = testing.iloc[:, :-1]
	Y_test = testing["κref."].astype("float64")

	global test_indices, train_indices
	test_indices = Y_test.index
	train_indices = Y_train.index

	return X_train, X_test, Y_train, Y_test

def preprocess_data_holdout_bottom(dataset):
	raw_X = dataset.iloc[:, 2:23]

	scaler = MinMaxScaler()
	scaler.fit(raw_X)

	X = pd.DataFrame(scaler.transform(dataset.iloc[:, 2:23]), columns=raw_X.columns)

	space_groups = dataset.loc[:, "space group"]
	dummies = pd.get_dummies(space_groups)
	X = pd.concat([dummies, X], axis=1)
	X.index = raw_X.index

	Y = dataset["κref."].astype("float64")

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80, shuffle=False)

	global test_indices, train_indices
	test_indices = Y_train.index
	train_indices = Y_test.index

	return X_test, X_train, Y_test, Y_train


def preprocess_data_holdout_random(dataset):
	raw_X = dataset.iloc[:, 2:23]

	scaler = MinMaxScaler()
	scaler.fit(raw_X)

	X = pd.DataFrame(scaler.transform(dataset.iloc[:, 2:23]), columns=raw_X.columns)

	# space_groups = dataset.loc[:, "space group"]
	# dummies = pd.get_dummies(space_groups)
	# X = pd.concat([dummies, X], axis=1)

	# X = dataset.loc[:, names]
	Y = dataset["κref."].astype("float64")

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, shuffle=True)

	global test_indices, train_indices
	test_indices = Y_test.index
	train_indices = Y_train.index

	return X_train, X_test, Y_train, Y_test

dataset = pd.read_csv("../dataset.csv", delimiter=",")
dataset_sorted = dataset.sort_values('κref.')

test_indices = None
train_indices = None

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
	X_train, X_test, Y_train, Y_test = preprocess_data_holdout_random(dataset)

raw_X = dataset.iloc[:, 2:23]

scaler = MinMaxScaler()
scaler.fit(raw_X)

X = pd.DataFrame(scaler.transform(dataset.iloc[:, 2:23]), columns=raw_X.columns)
space_groups = dataset.loc[:, "space group"]
dummies = pd.get_dummies(space_groups)
X = pd.concat([dummies, X], axis=1)

Y = dataset["κref."].astype("float64").values

slack_values = dataset["κ"].astype("float64").values
slack_test_values = None
slack_train_values = None
if test_indices is not None:
	slack_test_values = dataset.loc[test_indices, "κ"].astype("float64").values
	slack_train_values = dataset.loc[train_indices, "κ"].astype("float64").values

slack_mae = mean_absolute_error(Y, slack_values)
slack_r2 = r2_score(Y, slack_values)

models = []

#Plot the actual values versus the predicted values on a scatter plot
def plot(Y_actual, Y_generated, slack_values, r2_score, slack_r2_score, formula):
	fig, ax = plt.subplots()
	# actual_scatter = ax.scatter(range(len(dataset)), Y_actual, label="Actual")
	# generated_scatter = plt.scatter(range(len(dataset)), Y_generated, label="Generated")
	# slack_scatter = plt.scatter(range(len(dataset)), slack_values, label="Slack")
	# actual_scatter = ax.scatter(Y_actual, Y_actual, label="Observed")
	generated_scatter = plt.scatter(Y_generated, Y_actual, label="MLP", color="blue")
	slack_scatter = plt.scatter(slack_values, Y_actual, label="Slack", color="orange")
	# line = plt.plot([x + 1 for x in range(200)], range(200), color="black")
	line = plt.plot([x + 1 for x in list(range(0, math.ceil(max(Y_actual))))], list(range(0, math.ceil(max(Y_actual)))), color="black")
	ax.legend()
	plt.text(150, 20, "R² = " + str(round(r2_score, 3)), color="blue")
	plt.text(150, 00, "R² = " + str(round(slack_r2_score, 3)), color="orange")
	plt.suptitle(formula)
	plt.ylabel("Observed")
	plt.xlabel("Predicted")
	plt.title("MLP model versus slack model")
	plt.show()

def plot_sorted(Y_actual, Y_generated, slack_values, r2_score, slack_r2_score, formula):
	fig, ax = plt.subplots()
	# actual_scatter = ax.scatter(range(len(dataset)), Y_actual, label="Actual")
	# generated_scatter = plt.scatter(range(len(dataset)), Y_generated, label="Generated")
	# slack_scatter = plt.scatter(range(len(dataset)), slack_values, label="Slack")
	# actual_scatter = ax.scatter(Y_actual, Y_actual, label="Observed")
	generated_scatter = plt.scatter(Y_generated, Y_actual, label="MLP", color="blue")
	slack_scatter = plt.scatter(slack_values, Y_actual, label="Slack", color="orange")
	line = plt.plot([x + 1 for x in list(range(0, math.ceil(max(Y_actual))))], list(range(0, math.ceil(max(Y_actual)))), color="black")
	ax.legend()
	plt.text(150, 20, "R² = " + str(round(r2_score, 3)), color="blue")
	plt.text(150, 0, "R² = " + str(round(slack_r2_score, 3)), color="orange")
	plt.suptitle(formula)
	plt.ylabel("Observed")
	plt.xlabel("Predicted")
	plt.title("MLP model versus slack model on test values")
	plt.show()

def plot_train_test(Y_training_actual, Y_training_generated, Y_testing_actual, Y_testing_generated,
training_r2, testing_r2, training_rmse, testing_rmse):
	fig, ax = plt.subplots()
	train_label = "Train " + "(" + str(len(Y_training_actual)) + " points, " + "R²=" + str(round(training_r2, 3)) + ", RMSE=" + str(round(training_rmse, 3)) + ")"
	test_label = "Test " + "(" + str(len(Y_testing_actual)) + " points, " + "R²=" + str(round(testing_r2, 3)) + ", RMSE=" + str(round(testing_rmse, 3)) + ")"
	train_scatter = plt.scatter(Y_training_actual, Y_training_generated, label=train_label, color="orange")
	test_scatter = plt.scatter(Y_testing_actual, Y_testing_generated, label=test_label, color="blue", marker="s")
	line = plt.plot([x + 1 for x in list(range(0, math.ceil(max(Y))))], list(range(0, math.ceil(max(Y)))), "--", color="black")
	ax.legend()
	plt.xlabel("LTC Expected")
	plt.ylabel("LTC Predicted")
	plt.title("MLP model training vs testing")
	plt.show()

def plot_learning_curve(score_df):
	fig, ax = plt.subplots()
	folds = len(score_df)
	kfold_plot = plt.plot(range(1, folds + 1), score_df.loc[:, "K-Fold_Scores"], label="K-Fold MAE", color="orange", marker='o')
	holdout_plot = plt.plot(range(1, folds + 1), score_df.loc[:, "Holdout_Scores"], label="Holdout MAE", color="blue", marker='o')
	plt.xticks(range(1, folds + 1))
	ax.legend()
	plt.xlabel("Fold #")
	plt.ylabel("MAE of generated formula")
	plt.title("MLP model training vs testing")
	plt.show()

def plot_slack():
	slack_train_rmse = math.sqrt(mean_squared_error(Y_train, slack_train_values))
	slack_train_r2 = r2_score(Y_train, slack_train_values)
	slack_test_rmse = math.sqrt(mean_squared_error(Y_test, slack_test_values))
	slack_test_r2 = r2_score(Y_test, slack_test_values)

	fig, ax = plt.subplots()
	train_label = "Train " + "(" + str(len(Y_train)) + " points, " + "R²=" + str(round(slack_train_r2, 3)) + ", RMSE=" + str(round(slack_train_rmse, 3)) + ")"
	test_label = "Test " + "(" + str(len(Y_test)) + " points, " + "R²=" + str(round(slack_test_r2, 3)) + ", RMSE=" + str(round(slack_test_rmse, 3)) + ")"
	train_scatter = plt.scatter(Y_train, slack_train_values, label=train_label, color="orange")
	test_scatter = plt.scatter(Y_test, slack_test_values, label=test_label, color="blue", marker="s")
	line = plt.plot([x + 1 for x in list(range(0, math.ceil(max(Y))))], list(range(0, math.ceil(max(Y)))), "--", color="black")
	ax.legend()
	plt.suptitle("Slack-Berman")
	plt.xlabel("LTC Expected")
	plt.ylabel("LTC Predicted")
	plt.title("Slack model training vs testing")
	plt.show()

#Load the folders in the top_models folder, and sort them by ascending numerical value
try:
	folders = [x[1] for x in os.walk(model_dir)][0]
	folders_int = [[int(x), x] for x in folders]
	folders_int.sort()
	folders = [folders_int[x][1] for x in range(0, len(folders_int))]
except IndexError:
	print("There are no models currently saved in the top_models folder.")
else:
	for folder in folders:
		mlp = None
		try:
			mlp = load_model(model_dir + "/" + folder + "/model")

			holdout_data = pd.read_csv(model_dir + "/" + folder + "/holdout_data.csv")
			training_data = pd.read_csv(model_dir + "/" + folder + "/training_data.csv")
			score_df = pd.read_csv(model_dir + "/" + folder + "/learning_scores.csv", sep='\s*,\s*')
			models.append([mlp, holdout_data, training_data, score_df])

			predictions = mlp.predict(X)
			model_mae = mean_absolute_error(Y, predictions)
			model_r2 = r2_score(Y, predictions)

			if disp_testing:
				holdout_X_test = holdout_data.iloc[:, 1:-2].values
				holdout_Y_test = holdout_data.loc[:, "κref."].values
				predictions = mlp.predict(holdout_X_test)
				model_mae = mean_absolute_error(holdout_Y_test, predictions)
				model_r2 = r2_score(holdout_Y_test, predictions)
				model_rmse = math.sqrt(mean_squared_error(holdout_Y_test, predictions))
			else:
				predictions = mlp.predict(X)
				model_mae = mean_absolute_error(Y, predictions)
				model_r2 = r2_score(Y, predictions)
				model_rmse = math.sqrt(mean_squared_error(Y, predictions))

			if exclude_underperformers and (model_mae > slack_mae or model_r2 < slack_r2):
				continue
			else:
				print("Model " + folder + ": " + str(model_mae) + " : " + str(model_r2))
		except FileNotFoundError:
			print("Missing model #" + folder + ". Skipping...")
			continue

if disp_testing:
	print("Slack model:", mean_absolute_error(Y_test, slack_test_values), ":", r2_score(Y_test, slack_test_values))
else:
	print("Slack model:", slack_mae, ":", slack_r2)

if interactive:
	while True:
		print("\n")
		to_plot = input("Select a model to plot: ")

		if to_plot == "q" or to_plot == "quit" or to_plot == "exit" or to_plot == "":
			break
		elif to_plot.lower() == "slack model" or to_plot.lower() == "slack":
			plot_slack()
			continue
		else:
			to_plot = int(to_plot)

		training_dataset = models[to_plot][2]
		training_X = training_dataset.iloc[:, 1:-2].values
		training_Y = training_dataset.loc[:, "κref."].values

		score_df = models[to_plot][3]

		training_predictions = models[to_plot][0].predict(training_X)

		training_mae = mean_absolute_error(training_Y, training_predictions)
		training_r2 = r2_score(training_Y, training_predictions)
		training_rmse = math.sqrt(mean_squared_error(training_Y, training_predictions))

		holdout_data = models[to_plot][1]
		slack_test_values = holdout_data.loc[:, "κ"].values
		X_test = holdout_data.iloc[:, 1:-2].values
		Y_test = holdout_data.loc[:, "κref."].values
		test_predictions = models[to_plot][0].predict(X_test)
		testing_r2 = r2_score(Y_test, test_predictions)
		testing_rmse = math.sqrt(mean_squared_error(Y_test, test_predictions))

		plot_train_test(training_Y, training_predictions, Y_test, test_predictions, training_r2, testing_r2, training_rmse, testing_rmse)
		# plot_learning_curve(score_df)
