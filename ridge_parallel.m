%================================================
% Ridge regression with 10-fold cross validation
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
%   pval: p-value of the covariates
%   cv_results: the mse from cv of each lambda
%================================================
function [b,LRT_result,cv_results,mse,r_sq,best_lambda] = ridge_parallel(X,y,lambdas,opts)
        
    % set solver options
    if nargin < 3              
        error(message('Not enough inputs'));      
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

    [N,p]=size(X);
   
    % standardize the columns of input matrix X if requested
    if stdz == 1
        X = normalize(X);
    end


    % append a column of ones to X, make it a design matrix
    X = [ones(N,1),X];

    % split data into training and test set
    partition = cvpartition(N,'HoldOut',1-train_ratio);
    train_inds = training(partition);
    X_train = X(train_inds,:);
    y_train = y(train_inds);
    test_inds = test(partition);
    X_test = X(test_inds,:);
    y_test = y(test_inds);

    % compute cross-validation of the lambdas in parallel
    cv_results = zeros(length(lambdas),2);
    parpool(8);
    parfor i = 1:length(lambdas)
        lambda = lambdas(i);
        [cv_err]= cross_validation(X_train,y_train,lambda);
        cv_results(i,:)=[lambda,cv_err];
    end
    delete(gcp);
    % choose the lambda that gives smallest cross-validation error
    [err_min,ind_min] = min(cv_results(:,2));
    best_lambda = cv_results(ind_min,1);

    % fit the model with all training data with this lambda
    I = eye(p+1);
    b = (transpose(X_train)*X_train+best_lambda*I)\transpose(X_train)*y_train;

    % prediction using the resulting model on test set
    y_pred = X_test*b;
    residuals = y_pred-y_test;
    mse = mean(residuals.^2);
    
    % goodness-of-fit
    r_sq=1-(sum(residuals.^2)/sum((y_test-mean(y_test)).^2));

    % p-value using LRT
    %  log-likelihood of full model
    logL_full = compute_loglikelihood(y_pred,y_test);

    % fit reduced model, parallely
    LRT_result = zeros(p,4);
    parpool(8);

    parfor i = 2:(p+1)
        X_rm_train = X_train(:,[1:i-1,i+1:end]);
        X_rm_test = X_test(:,[1:i-1,i+1:end]);
        I = eye(p);
        b_rm = (transpose(X_rm_train)*X_rm_train+best_lambda*I)\transpose(X_rm_train)*y_train;
        y_rm_pred=X_rm_test*b_rm;
        [h,pval,stat] = lratiotest(logL_full,compute_loglikelihood(y_rm_pred,y_test),1);
        LRT_result(i-1,:)=[i-1,h,pval,stat];
    end
    delete(gcp);

end

