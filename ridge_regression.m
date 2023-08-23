%================================================
% Initiation of ridge regression
% Input:
%   X: the Nxp input matrix with N observations on p coefficients
%   y: the N*1 observation matrix
%   lambdas: the list of lambdas for regularization, e.g. exp(-8:6)
%   train_ratio: proportion of data used for training, default 0.8
%
%  OUTPUT:
%   b: coefficient estimates
%   LRT_result: results from likelihood-ratio test
%       includes the i-th coefficient (except the constant term), 
%                   reject null hypothesis or not, 
%                   p-value
%                   test statistic value (chi-squared test)
%   mse: the mse of the model tested on the test set
%   r_sq: r-square, goodness of fit
%================================================
function [b,LRT_result,mse,r_sq] = ridge_regression(X,y,lambdas,train_ratio)

    if nargin < 2              
        error('Not enough inputs');      
    end

    % non-numerical entries are not supported yet
    if ~isnumeric(X)||~isnumeric(y)
        error('Data entries must be numeric');
    end

    [nx,p]=size(X);
    ny=length(y);
    if nx~=ny
        error('Sample sizes of input and output matrix do not match');
    end
    
    if ~exist('lambdas','var')
        lambdas=exp(-8:6);
    end
    if ~isnumeric(lambdas)
        error('Ridge tuning parameter lambda must be numeric');
    end

    if ~exist('train_ratio','var')
        train_ratio=0.8;
    end

    if train_ratio<0.7
        error('Training set should be at least 70% of the whole data');
    elseif train_ratio>=1
        error('Training set cannot be greater than 100% of the whole data');
    elseif ~isnumeric(train_ratio)
        error('Training ratio must be numeric');
    end

    % remove missing values in X and y
    nas_ind = (isnan(y) | any(isnan(X),2));
    if any(nas_ind)
        X = X(~nas_ind,:);
        y = y(~nas_ind);
    end
    fprintf('Removed %d samples due to missing values\n',sum(nas_ind));

    if p<999
        [b,LRT_result,mse,r_sq] = ridge_nopar(X,y,lambdas,train_ratio);
    else
        [b,LRT_result,mse,r_sq] = ridge_parallel(X,y,lambdas,train_ratio);
    end

end