%================================================
% Initiation of ridge regression
% Input:
%   X: the Nxp input matrix with N observations on p coefficients
%   y: the N*1 observation matrix
%   lambdas: the list of lambdas for regularization
%   opts: (Optional) 1x1 struct including solver options
%         standardize: to standardize the input X or not, enter 0 or 1,
%           default is 0
%         train_ratio: proportion of data used for training, default 0.8
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
function [b,LRT_result,mse,r_sq] = ridge_regression(X,y,lambdas,opts)

    if nargin < 3              
        error(message('Not enough inputs'));      
    end

    % non-numerical entries are not supported yet
    if ~isnumeric(X)||~isnumeric(y)
        error(message('Data entries must be numeric'));
    end
    if ~isnumeric(lambdas)
        error(message('Ridge tuning parameter lambda must be numeric'));
    end

    [nx,p]=size(X);
    ny=length(y);
    if nx~=ny
        error(message('Sample sizes of input and output matrix do not match'));
    end

    if (nargin<4)||isempty(opts)||~isfield(opts,'standardize')
        stdz=0;
    else
        stdz=1;
    end
    
    if (nargin<4)||isempty(opts)||~isfield(opts,'train_ratio')
        train_ratio=0.8;
    else
        if opts.train_ratio<0.7
            error(message('Training set should be at least 70% of the whole data'));
        else
            train_ratio=opts.train_ratio;
        end
    end

    % remove missing values in X and y
    nas_ind = (isnan(y) | any(isnan(X),2));
    if any(nas_ind)
        X = X(~nas_ind,:);
        y = y(~nas_ind);
    end
    fprintf('Removed %d samples due to missing values\n',sum(nas_ind));

    if p<999
        [b,LRT_result,mse,r_sq] = ridge_nopar(X,y,lambdas,stdz,train_ratio);
    else
        [b,LRT_result,mse,r_sq] = ridge_parallel(X,y,lambdas,stdz,train_ratio);
    end

end