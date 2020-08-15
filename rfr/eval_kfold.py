import sys

interactive = False
flag_missing_vars = False
exclude_underperformers = False
disp_testing = False
bottom_eval = False
top_bottom_eval = False
top_eval = False
true_middle_eval = False
use_rmse = False

model_dir = "top_models_kfold_cv"

if "-i" in sys.argv or "--interactive" in sys.argv:
	interactive = True
if "-h" in sys.argv or "--help" in sys.argv:
	print("LTC Symbolic Regression Help Menu")
	print("Usage: python eval.py [arguments]")
	print("Options/arguments:")
	print("     -h or --help: Displays this menu")
	print("     -i or --interactive: Launch in interactive mode. Allows for plotting and displaying of equations interactively.")
	print("     -e or --exclude: Omits models that did not outperform the Slack Berman model when scored against the entire dataset")
	print("     -r or --use-rmse: Displays RMSE instead of MAE")
	print("     -t or --testing: Displays testing MAE and R2 insead of average MAE and R2")
	sys.exit()
if "-f" in sys.argv or "--flag" in sys.argv:
	flag_missing_vars = True
if "-e" in sys.argv or "--exclude" in sys.argv:
	exclude_underperformers = True
if "-t" in sys.argv or "--testing" in sys.argv:
	disp_testing = True
if "-r" in sys.argv or "--use-rmse" in sys.argv:
	use_rmse = True

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import math
import pickle


def load_model(filepath):
	with open(filepath, 'rb') as file:
		return pickle.load(file)

def preprocess_space_groups(dataset):
	space_groups = dataset.loc[:, "space group"]
	labels, uniques = pd.factorize(space_groups)
	dataset["space group"] = labels
	return dataset.drop("Unnamed: 1", axis=1)

dataset = pd.read_csv("../dataset.csv", delimiter=",")
dataset = preprocess_space_groups(dataset)
dataset_sorted = dataset.sort_values('κref.')

X = dataset.iloc[:, 2:23]
Y = dataset["κref."].astype("float64")

slack_values = dataset.loc[:, "κ"]
slack_mae = mean_absolute_error(Y, slack_values)
slack_r2 = r2_score(Y, slack_values)

X_train = pd.DataFrame().reindex_like(X)
X_test = pd.DataFrame().reindex_like(X)
Y_train = pd.Series()
Y_test = pd.Series()

X_train = X_train.iloc[0:0]
X_test = X_test.iloc[0:0]

slack_train_values = pd.Series()
slack_test_values = pd.Series()

var_names = X.columns
X = X.values
Y = Y.values

variables = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]

models = []

def simplify(formula):
	s=formula.replace("numpy_protected_sqrt","sqrt")
	s=s.replace("numpy_protected_div_one", "÷")
	s=s.replace("numpy_protected_log_abs","log_abs").replace("subtract","-").replace("multiply",'*').replace("add","+")
	s=s.replace("cbrt","root3").replace("true_divide","÷")

	for i in range(len(var_names)):
		s = s.replace("X" + str(i), var_names[i])
		s = s.replace("T" + str(i), var_names[i])
	return s

#Plot the actual values versus the predicted values on a scatter plot
def plot(Y_actual, Y_generated, model_rmse, model_r2):
	fig, ax = plt.subplots()
	label_str = str(len(Y_actual)) + " points, " + "R²=" + str(round(model_r2, 3)) + ", RMSE=" + str(round(model_rmse, 3))
	scatter = plt.scatter(Y_actual, Y_generated, label=label_str, color="orange")
	line = plt.plot([x + 1 for x in list(range(0, math.ceil(max(Y))))], list(range(0, math.ceil(max(Y)))), "--", color="black")
	ax.legend()
	plt.xlabel("LTC Expected")
	plt.ylabel("LTC Predicted")
	plt.title("RFR model")
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

def plot_learning_curve(score_df):
	fig, ax = plt.subplots()
	folds = len(score_df)
	kfold_plot = plt.plot(range(1, folds + 1), score_df.loc[:, "K-Fold_Scores"], label="K-Fold MAE", color="orange", marker='o')
	holdout_plot = plt.plot(range(1, folds + 1), score_df.loc[:, "Holdout_Scores"], label="Holdout MAE", color="blue", marker='o')
	plt.xticks(range(1, folds + 1))
	ax.legend()
	plt.xlabel("Fold #")
	plt.ylabel("MAE of generated formula")
	plt.title("RFR model training vs testing")
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
		rfr = RandomForestRegressor()
		try:
			rfr = load_model(model_dir + "/" + folder + "/model")
			missing_var = False

			training_data = pd.read_csv(model_dir + "/" + folder + "/training_data.csv")
			test_data = pd.read_csv(model_dir + "/" + folder + "/test_data.csv")
			score_df = pd.read_csv(model_dir + "/" + folder + "/learning_scores.csv", sep='\s*,\s*')
			models.append([rfr, test_data, training_data, score_df])

			if disp_testing:
				test_X_test = test_data.iloc[:, 1:-2].values
				test_Y_test = test_data.loc[:, "κref."].values
				predictions = rfr.predict(test_X_test)
				model_mae = mean_absolute_error(test_Y_test, predictions)
				model_r2 = r2_score(test_Y_test, predictions)
				model_rmse = math.sqrt(mean_squared_error(test_Y_test, predictions))
			else:
				predictions = rfr.predict(X)
				model_mae = mean_absolute_error(Y, predictions)
				model_r2 = r2_score(Y, predictions)
				model_rmse = math.sqrt(mean_squared_error(Y, predictions))
			if exclude_underperformers and (model_mae > slack_mae or model_r2 < slack_r2):
				continue
			else:
					if missing_var and flag_missing_vars:
						print("!", end="")
					if use_rmse:
						print("Model " + folder + ": " + str(model_rmse) + " : " + str(model_r2))
					else:
						print("Model " + folder + ": " + str(model_mae) + " : " + str(model_r2))
		except FileNotFoundError:
			print("Missing model #" + folder + ". Skipping...")
			continue

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
		# training_dataset['V'].replace('', np.nan, inplace=True)
		# training_dataset.dropna(subset=['V'], inplace=True)

		training_Y = training_dataset.loc[:, "κref."].values

		score_df = models[to_plot][3]

		Y_test = test_data.loc[:, "κref."].values
		slack_train_values = training_dataset.loc[:, "κ"].values
		slack_mae = mean_absolute_error(training_Y, slack_train_values)
		slack_r2 = r2_score(training_Y, slack_train_values)

		if disp_testing:
			print("Slack model:", mean_absolute_error(Y_test, slack_test_values), ":", r2_score(Y_test, slack_test_values))
		else:
			print("Slack model:", slack_mae, ":", slack_r2)

		predictions = models[to_plot][0].predict(X)
		model_rmse = math.sqrt(mean_squared_error(Y, predictions))
		model_r2 = r2_score(Y, predictions)

		plot(Y, predictions, model_rmse, model_r2)
		# plot_learning_curve(score_df)
		# print("MAE on full set: " + str(model_mae))
		# print("Slack MAE on full set: " + str(slack_mae))
		# print("")
		# print("R2 on full set: " + str(model_r2))
		# print("Slack R2 on full set: " + str(slack_r2))
		# plot(Y, predictions, slack_values, model_r2, slack_r2, models[to_plot][1])

		# print("---------------------")

		# slack_test_mae = mean_absolute_error(Y_test, slack_test_values)
		# slack_test_r2 = r2_score(Y_test, slack_test_values)
		# test_mae = mean_absolute_error(Y_test, test_predictions)
		# test_r2 = r2_score(Y_test, test_predictions)
		# print("MAE on test values: " + str(test_mae))
		# print("Slack MAE on test values: " + str(slack_test_mae))
		# print("")
		# print("R2 on test values: " + str(test_r2))
		# print("Slack R2 on test values: " + str(slack_test_r2))
		# plot(Y_test, test_predictions, slack_test_values, test_r2, slack_test_r2, models[to_plot][1])
		# training_preds = models[to_plot][0].predict(X_sorted_train)
		# testing_preds = models[to_plot][0].predict(X_sorted_test)
