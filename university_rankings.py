# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ## Objective - Rank universities based on different features
# %% [markdown]
# #### Import modules and prepare dataset.

# %%
# Import sklearn tools
from sklearn.svm import SVR
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Import other libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from scipy.special import expit
from mlxtend.plotting import scatterplotmatrix
get_ipython().run_line_magic('matplotlib', 'inline')

# Load dataset
dataset = pd.read_csv('shanghai_data.csv')

# Check which columns have null values.
for col in dataset.columns:
    pct_missing = np.mean(dataset[col].isnull())
    print(f"% of missing data in '{col}' - {pct_missing*100:.2f}%")

# Fill the data.
dataset.fillna(dataset.mean(), inplace=True)

# Checking for duplicates. NOTE: world_rank will become the target.
print('Duplication in dataset: {}'.format(dataset.duplicated(subset=['world_rank']).any()))    # -> True

# NOTE: There are duplicates in the dataset because the rankings change every year, so the rankings are updated.

# Split data into target and data variables.
data_df = dataset.loc[:, 'total_score':'pcp']

# Encoding target colum (target is not continuous).
target_df = dataset.loc[:, 'world_rank']

data = data_df.to_numpy()
target = target_df.to_numpy()

print('data.shape: {}'.format(data.shape))
print('target.shape: {}'.format(target.shape))

# University names (for reference).
university_names = dataset.loc[:, 'university_name'].to_numpy()

print(f'target.dtype: {target.dtype}')    # -> object

# NOTE: The reason why the above code returns object is bc the data is string,
#       but the strings must be the same length, so it is stored in this dtype.

# Get indexes where the targets can be converted to floats.
indexes = []

for index in range(len(target)):
    if '-' not in target[index]:
        indexes.append(index)

print('target (after sample selection): {}'.format(target := target[indexes].astype(float)))

print('target.shape (after sample selection): {}'.format(target.shape))
print('data.shape (after sample selection): {}'.format((data := data[indexes]).shape))

# Split the data.
X_train, X_test, y_train, y_test = train_test_split(data, target)

# Scale the data.
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f'X_train.shape: {X_train.shape}')
print(f'y_train.shape: {y_train.shape}')
print(f'X_test.shape: {X_test.shape}')
print(f'y_test.shape: {y_test.shape}')

# %% [markdown]
# #### Data analysis

# %%
# Check if there are a significant amount of zeros in the data.
print(f'% of zeros in data -> {np.sum(data == 0)/data.size}')
print(f'% of nonzeros in data -> {np.sum(data != 0)/data.size}')

# Check how each feature is correlated to the target.
fig, axs = plt.subplots(2, 4, figsize=(11, 5.5))
features = ['total_score', 'alumni', 'award', 'hici', 'ns', 'pub', 'pcp']

axs = [ax for ax in axs.ravel()]

for i in range(7):
    axs[i].scatter(data[:, i], target)
    axs[i].set_title(features[i])

# NOTE: total_score seems like the most correlated feature.
# NOTE: It seems as if all of the features do not have a linear relationship towards the target variable.
#       This means that a polynomial regression model would be preferred.

# %% [markdown]
# #### Apply model to the data.

# %%
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10]
}

gs_params = {
    'estimator': SVR(kernel='linear'),
    'param_grid': param_grid,
    'n_jobs': -1,
}

gs = GridSearchCV(**gs_params)
gs.fit(X_train, y_train)

svr = gs.best_estimator_

# %% [markdown]
# #### View the results of the algorithm

# %%
print(f'Algorithm: {svr}')

pred_train = svr.predict(X_train)
pred_test = svr.predict(X_test)

print(f'Train MSE: {mean_squared_error(pred_train, y_train)}')
print(f'Test MSE: {mean_squared_error(pred_test, y_test)}')

# NOTE: anything below 500 is an ideal MSE.

# Show predictions for 5 random schools.

indexes = set()

for i in range(5):
    indexes.add(choice(range(len(X_test))))

for index in indexes:
    print(f'\n----- {university_names[index]} -----')
    print(f'\tPrediction: {svr.predict(X_test[index].reshape(1, -1))[0]:.2f}')
    print(f'\tTarget: {y_test[index]}')

# %% [markdown]
# #### Plot the results

# %%



