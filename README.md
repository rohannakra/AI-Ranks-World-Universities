# AI Ranks Universities

Steps taken:
* prepare dataset
    * change target variable to ```float```
    * scale the data using ```StandardScaler()```
* data analysis
    * check # of zeros in the data
    * plot the data to check correlation of features to ```target``` variable
* apply model to the data
* view the results of the algorithm
    * print predictions for 5 random schools.
* plot the results
	* use ```coef_``` and ```intercept_``` attributes to plot deciscion boundary