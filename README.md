# AI Ranks Universities

Steps taken:
* prepare dataset
    * change target variable to ```float```
    * scale the data using ```StandardScaler()```
* data analysis
    * check # of zeros in the data
    * plot the data
* apply model to the data
    * use ```GridSearchCV``` to find best parameters
* view the results of the algorithm
    * print the ```mean_squared_error()```
    * print predictions for 5 random schools.
* plot the results
	* use ```coef_``` and ```intercept_``` attributes to plot deciscion boundary