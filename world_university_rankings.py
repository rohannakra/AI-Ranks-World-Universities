# Objective: Rank universities based on a # of features

# Import sklearn tools
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from IPython.display import display

# Import other libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect
from random import choice
import os

# Getting workspace so I can re-route the path.
print(f'Workspace: {os.getcwd()}')

dataset = pd.DataFrame(pd.read_csv(os.path.join('Projects', 'World University Rankings', 'shanghai_data.csv')))

# Check which columns have null values.
for col in dataset.columns:
    pct_missing = np.mean(dataset[col].isnull())
    print(f"Missing data % of '{col}' - {pct_missing*100:.2f}%")

# Dropping columns and samples.
dataset.drop(['national_rank', 'year'], axis=1, inplace=True)
dataset.dropna(inplace=True)

# Checking for duplicates. NOTE: world_rank will become the target.
print(dataset.duplicated(subset=['world_rank']).any())    # -> True

# NOTE: There are duplicates in the dataset because the rankings change every year, so the rankings are updated.

for col in dataset.columns:
    pct_missing = np.mean(dataset[col].isnull())
    print(f"Checking % of missing data in '{col}' - {pct_missing*100:.2f}%")


# Split data into target and data variables.]
data = dataset.loc[:, 'total_score':].to_numpy()
# Encoding target colum (target is not continuous).
target = dataset.loc[:, 'world_rank'].to_numpy()
# University names.
university_names = dataset.loc[:, 'university_name'].to_numpy()

# Found out that the target variables were strings, so I'm going to change it to numbers.
print(f'Checking what data type target variables are {np.unique(target)}')

# NOTE: target is specifically numbers 1 - 100. This can lead for the model to overfit.
#       I will encode the variables so that they look like ('1-5').

boundaries = [6, 16, 26, 51, 76]
target = np.array([bisect.bisect_left(boundaries, int(t)) for t in target])
print(f'Checking the data of target after transformation {target}')
print(np.unique(target))

target_names = ['1 - 5', '6 - 15', '16 - 25', '26 - 50', '51 - 75', '76 - 100']

print(f'new unique values {np.unique(target)}')

print(f'data.shape -> {data.shape}')
print(f'data.size -> {data.size}')
print(f'target.shape -> {target.shape}')

# Check if there are a significant amount of zeros in the data.
print(f'% of nonzeros in data -> {np.sum(data == 0)/data.size}')
print(f'% of nonzeros in data -> {np.sum(data != 0)/data.size}')

# -----------------------------------------------------------------------------------------------

# Objective: Do some data exploration.

tsne = TSNE(random_state=42)

data_trans = tsne.fit_transform(data)

scatter_plot = plt.scatter(data_trans[:, 0], data_trans[:, 1], c=target, label=target_names)
plt.legend(*scatter_plot.legend_elements())
plt.show()

# NOTE: Found out there was linearity in data, I will use models which
#       work well with linear data, such as neural networks or svms.
#       Non-Linear models such as ensemble methods or SVM(kernel='poly')

# ----------------------------------------------------------------------------------------------------

# Objective: Apply MLPClassifier to the data.

# Arguments for algorithm.
mlp_args = {
    'hidden_layer_sizes': (1000,),
    'solver': 'adam',
    'max_iter': 500,
    'alpha': 0.01
}

# Model to feed into cross_validate loop.
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(**mlp_args))
])

# Arguments for cross validation.
cross_validate_args = {
    'cv': 5,
    'n_jobs': -1,
    'verbose': 100,
    'return_train_score': True,
    'return_estimator': True
}

# Apllying cross_validate() to neural network.
mlp = cross_validate(pipe, data, target, **cross_validate_args)

# Making cross validation on mlp without scaler.
mlp_wo_scale = cross_validate(MLPClassifier(**mlp_args), data, target, **cross_validate_args)

# Creating scores variables.
mlp_scores = {
    'train': np.average(mlp['train_score']),
    'test': np.average(mlp['test_score'])
}

mlp_wo_scale_scores = {
    'train': np.average(mlp_wo_scale['train_score']),
    'test': np.average(mlp_wo_scale['test_score'])
}

print('Train Score w/ scaler: {} Train Score w/out: {}'.format(mlp_scores['train'], mlp_wo_scale_scores['train']))
print('Test Score w/ scaler: {} Test Score w/out: {}'.format(mlp_scores['test'], mlp_wo_scale_scores['test']))

# ---------------------------------------------------------------------------------------------------------------------

# Objective: Apply linear model to dataset.

svm = cross_validate(SVC(kernel='linear'), data, target, **cross_validate_args)
displayed_results = pd.DataFrame(svm)

print(displayed_results.loc[:, 'test_score':])

svm_scores = {
    'train': np.average(svm['train_score']),
    'test': np.average(svm['test_score'])
}

print(svm_scores['train'])
print(svm_scores['test'])

#  -------------------------------------------------------------------------

# Objective - Visualize results.

if mlp_scores['test'] > svm_scores['test']:
    best_clf = mlp['estimator'][0]    # This will return the pipeline instead of just the MLP.
else:
    best_clf = svm['estimator'][0]

# Checking which estimator had the best test score.
print(best_clf.__class__.__name__)    # -> Pipeline
print(best_clf)

# Get predictions from best_clf.
predictions = cross_val_predict(best_clf, data, target, cv=5, n_jobs=-1, verbose=100)

# predictions==target will compare the predictions to target -> [True True True...].
# changing the data type will make the True -> 1 and False -> 0.
# This is for the program to color coordinate the predictions.
bin_pred = (predictions==target).astype('int32')
print(bin_pred)    # -> [1 1 1 ... 1 1 1]

# NOTE: cross_val_predict returns the estimator predicting y_train
print(np.sum(bin_pred)/len(bin_pred))    # Checking if score is the same.
print(mlp_scores['test'])    # Mostly...

# Check how many times the model predicted correctly.
print(np.bincount(bin_pred))

fig, (ax_1, ax_2) = plt.subplots(1, 2, subplot_kw={'yticks': (), 'xticks': ()}, figsize=(18, 9))

ax_1.scatter(data_trans[:, 0], data_trans[:, 1])
scatter_pred = ax_2.scatter(data_trans[:, 0], data_trans[:, 1], c=bin_pred, label=bin_pred)

ax_2.legend(*scatter_pred.legend_elements())

ax_1.set_title('Labels')    # TODO: Fix.
ax_2.set_title('Predictions')

plt.show()

# ---------------------------------------------------------------------------------------------

# Objective - Show predictions for 25 random schools.

# Creating list to mutate.
target_indices = list(range(len(target)))
indices = []

for i in range(25):
    indices.append(choice(target_indices))

# Get rid of coppies.
indices = set(indices)

for index in indices:
    print(f'\n----- {university_names[index]} -----\n Prediction -> {target_names[predictions[index]]}\n Target -> {target_names[target[index]]}\n')
