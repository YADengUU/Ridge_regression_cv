
# Ridge Regression with Cross Validation

## Project Description

+ Performs Ridge regression for linear models ($y=X\beta+\varepsilon, \varepsilon \sim N(0,\sigma^2)$) with 10-fold cross validation
	Compared to normal equation of solving linear models
		$\hat{\beta}=(X^T X)^{-1} X^T y$
	Ridge regression adds L2 penalty term to ensure invertibility
		$\hat{\beta}=(X^T X + \lambda I)^{-1} X^T y$
+ The built-in Ridge regression function in MATLAB ridge(y,X,k) does not have cross-validation to select the $\lambda$ that gives smallest MSE (mean squared error). It only computes the estimation based on the given penalty value k.

+ While our ridge_regression(X, y) has default list of lambdas exp(-8:6), it also enables users to provide a list of $\lambda$ to be tested and finally computes the coefficient estimates using the $\lambda$ which gives the smallest MSE across all folds. User may also specify an additional input for the proportion of data used for training (default is 80%, i.e., 0.8) by ridge_regression(X, y, lambdas, trainRatio). To include the two customized inputs, one can do ridge_regression(X, y, exp(-5,5), 0.75). 

+ In addition to the coefficient estimates $\beta$s, the outputs of ridge_regression has the MSE of the fitted model on test set as well as r^2 indicating the goodness of fit, and results from likelihood ratio test. One may check the p-value and significance by viewing the results of the likelihood ratio test. For example, by running 
 [b,LRT_result,mse,r_sq] = ridge_regression(X,y)

+ user can view the coefficient estimates in b, the MSE and r^2 in mse and r_sq. LRT_results is a matrix with rows as [coefficient, reject_null, p_value, test_statistic] for each coefficient except the constant term.

## Caution

The current version does not support non-numerical, i.e., qualitative entries.

## Installation

The scripts can be downloaded to run in MATLAB (version 2022b or 2023a). 

Or run the command line:
```
docker run -it --rm --shm-size=512M ridge_regression_project
```
which prompts user to enter the MATLAB account and password before being able to run the function ridge_regression(X, y) or ridge_regression(X, y,lambdas,train_ratio) if customized \lambda’s and training ratio is desired.




