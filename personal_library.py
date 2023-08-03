# this is the library I made for the credit score case
# there are a lot of redundancies in the functions, which I resolved
# by implement OOP with classes and such, but I noticed a slight
# decrease in running time, so I stuck with this version instead
# prioritizing efficiency over clean code

import pandas as pd
import numpy as np

def roc(false_positive, true_positive):
	import matplotlib.pyplot as plt
	plt.plot(false_positive, true_positive)
	plt.xlabel('False Positive rt.')
	plt.ylabel('True Positive rt.')
	plt.title('Receiver Operating Characteristic (ROC) curve for CV')
	x = np.linspace(0, 1, 1000)
	plt.plot(x, x)
	plt.show()

def linear_regression(data, predicted_column):
	from sklearn.metrics import mean_squared_error
	from sklearn.model_selection import train_test_split
	from keras.models import Sequential as seq
	from keras.layers import Dense as den
	data_columns = data.columns

	#define the predictors and the target column
	predictors = data[data_columns[data_columns != predicted_column]] #here is the target
	target = data[predicted_column]

	#normalize predictors
	predictors_normalized = (predictors - predictors.mean())/ predictors.std()

	#this defines the predictors (X) and target (y) for training and testing
	X_train, X_test, y_train, y_test =  train_test_split(predictors, target, test_size = 0.3, random_state = 42)

	#predictors_stdr = (predictors - predictors.mean()) / predictors.std() #here we standardize the predictors

	n_columns = predictors.shape[1]
	n_classes = y_test.shape[0]

	#with this funciton we define the model and add the layers to it
	#with posterior use of the function compile, which makes the regression possible
	def regression_model():
		model = seq()
		model.add(den(10, activation = 'relu', input_shape = (n_columns,)))
		model.add(den(5, activation = 'relu'))
		model.add(den(1))

		model.compile(optimizer = 'adam', loss = 'mean_squared_logarithmic_error')

		return model

	#this builds the model
	model = regression_model()

	mse_list = []
	#this trains and predicts
	for i in range(1, 3):
		model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 30, batch_size = 200, verbose = 1)
		y_pred = model.predict(X_test)
		mserr = mean_squared_error(y_test, y_pred[:,0])
		mse_list.append(mserr)  

	#and this evaluates it
	mean, std_dev = np.mean(mse_list), np.std(mse_list)
	print('Mean = {:.2f}, Std = {:.2f}'.format(mean, std_dev))
	return model

def random_forest(df, predictor):
	from sklearn.model_selection import train_test_split
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.preprocessing import StandardScaler
	import sklearn.metrics as metrics

	#for the randomforest, we convert the dataframe first
	label = np.array(df[predictor])
	features = df.drop(predictor, axis = 1)
	column_list = np.array(list(features.columns))

	train, test, train_labels, test_labels =  train_test_split(features, label, test_size = 0.3, random_state = 42)
	scaler_ = StandardScaler().fit(train)
	train_scaled = scaler_.transform(train)
	test_scaled = scaler_.transform(test)


	rfc = RandomForestClassifier(n_estimators = 300, random_state = 42)
	rfc.fit(train_scaled, train_labels)
	prediction_column = rfc.predict_proba(test_scaled)
	#prediction_probability = rfc.predict_proba(test)
	#for i,v in enumerate(rfc.feature_importances_):
	#print('Feature: %0d, Score: %.5f' % (i,v))
	#error = abs(prediction_column - test_labels)
	#print('Error abs: ', error) 

	#this is for the classifier validation
	clf_scores = prediction_column[:,1]
	fpr, tpr, threshold = metrics.roc_curve(test_labels, clf_scores)
	auc = metrics.roc_auc_score(test_labels,clf_scores)
	print('AUC Score : ', (auc))
	roc(fpr, tpr)
	return {'Model' : rfc, 'AUC': auc}

def logit_regression(df, predictor):
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
	from sklearn.preprocessing import StandardScaler
	import sklearn.metrics as metrics

	label = np.array(df[predictor])
	features = df.drop(predictor, axis = 1)

	train, test, train_labels, test_labels =  train_test_split(features, label, test_size = 0.3, random_state = 42)
	scaler_ = StandardScaler().fit(train)
	train_scaled = scaler_.transform(train)
	test_scaled = scaler_.transform(test)

	lgr = LogisticRegression(random_state=123, solver='saga', penalty='l1', class_weight='balanced', C=1.0, max_iter=500)

	lgr.fit(train_scaled, train_labels)
	predictions = lgr.predict_proba(test_scaled)[:,1]  
	fpr, tpr, threshold = metrics.roc_curve(test_labels, predictions)
	roc_auc = metrics.auc(fpr, tpr)
	auc = round(metrics.roc_auc_score(test_labels, predictions), 7)
	print('AUC Score model: ', auc)

	#this is for cross validation
	#clf = LogisticRegressionCV(Cs=[1, 2, 5, 10], random_state=123, penalty='l1', solver='saga', class_weight='balanced', max_iter=500) # If Cs not specified, default=10.
	#clf_scores = lgr_scores[:,1]
	#fpr, tpr, threshold = metrics.roc_curve(test_labels, clf_scores)
	#auc_cv = metrics.roc_auc_score(test_labels,clf_scores)
	#print('AUC Score CV: ', (auc_cv))
	roc(fpr, tpr)	

	return {'Model' : lgr, 'ModelPredictions' : predictions, 'AUC': auc}

def feature_permutation(df, predictor, model):
	from sklearn.inspection import permutation_importance
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	import sklearn.metrics as metrics

	label = np.array(df[predictor])
	features = df.drop(predictor, axis = 1)

	train, test, train_labels, test_labels =  train_test_split(features, label, test_size = 0.3, random_state = 42)
	scaler_ = StandardScaler().fit(train)
	train_scaled = scaler_.transform(train)
	test_scaled = scaler_.transform(test)

	pmi = permutation_importance(
	model, test_scaled, test_labels, n_repeats=15, random_state=42, n_jobs=2)

	feature_names = [f"feature {i}" for i in range(features.shape[1])]

	pmi_importances = pd.Series(pmi.importances_mean, index=feature_names)

	return {'Model': pmi, 'Importances': pmi_importances}

def gradient_boosting(df, predictor):
	from sklearn import datasets, ensemble
	import sklearn.metrics as metrics
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler

	label = np.array(df[predictor])
	features = df.drop(predictor, axis = 1)

	train, test, train_labels, test_labels =  train_test_split(features, label, test_size = 0.3, random_state = 42)
	scaler_ = StandardScaler().fit(train)
	train_scaled = scaler_.transform(train)
	test_scaled = scaler_.transform(test)

	gb = ensemble.GradientBoostingClassifier(n_estimators = 200, max_depth = 4, min_samples_split = 5, learning_rate = 0.04)
	gb.fit(train_scaled, train_labels)
	gb_importances = gb.feature_importances_
	prediction_column = gb.predict_proba(test_scaled)[:,1]  
	auc = round(metrics.roc_auc_score(test_labels, prediction_column), 7)
	fpr, tpr, threshold = metrics.roc_curve(test_labels, prediction_column)
	roc(fpr, tpr)	
	print(threshold)
	print('AUC Score model: ', auc)

	return {'Model': gb, 'Importances': gb_importances, 'AUC' : auc}

def XG_boosting(df, predictor):
	from sklearn import datasets, ensemble
	import sklearn.metrics as metrics
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	import xgboost as Xgb

	label = np.array(df[predictor])
	features = df.drop(predictor, axis = 1)

	train, test, train_labels, test_labels =  train_test_split(features, label, test_size = 0.3, random_state = 42)
	scaler_ = StandardScaler().fit(train)
	train_scaled = scaler_.transform(train)
	test_scaled = scaler_.transform(test)

	xgb = Xgb.XGBClassifier(n_estimators = 200, max_depth = 4, min_samples_split = 5, learning_rate = 0.04)
	xgb.fit(train_scaled, train_labels)
	xgb_importances = xgb.feature_importances_
	prediction_column = xgb.predict_proba(test_scaled)[:,1]  
	auc = round(metrics.roc_auc_score(test_labels, prediction_column), 7)
	fpr, tpr, threshold = metrics.roc_curve(test_labels, prediction_column)
	roc(fpr, tpr)
	print('AUC Score model: ', auc)

	return {'Model': xgb, 'Importances': xgb_importances, 'AUC' : auc}
