%================================================
% Ridge regression with 10-fold cross validation (without parallelization)

function [b_ols,mse_ols,r_sq_ols] = ols_estimator(X,y,train_ratio)
    
    

    [N,~]=size(X);

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
    
    b_ols=(transpose(X_train)*X_train)\transpose(X_train)*y_train;

    y_ols = X_test*b_ols;
    mse_ols=mean((y_ols-y_test).^2);
    r_sq_ols=1-(sum((y_ols-y_test).^2)/sum((y_test-mean(y_test)).^2));

    
end

