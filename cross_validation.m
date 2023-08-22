function [cv_err]=cross_validation(X,y,lambda)
    % 10-fold cross validation,here X and y_obs are the training sets
    %  this is for selecting the best lambda, function is looped for lambdas

    [N,p] = size(X); % X here has the column of constant term
    numFolds = 10;

    cv_inds = crossvalind('Kfold',N,numFolds);

    % store evaluation metrics
    errs = zeros(numFolds,1);

    for fold = 1:numFolds
        test_inds = (cv_inds==fold);

        X_test = X(test_inds,:);
        y_test = y(test_inds);

        train_inds = (~test_inds);
        X_train = X(train_inds,:);
        y_train = y(train_inds);
        
        % compute b
        I = eye(p);
        b = (transpose(X_train)*X_train+lambda*I)\transpose(X_train)*y_train;

        % prediction by the resulting model
        y_pred = X_test*b;
        
        % compute MSE of current fold
        mse = mean((y_pred-y_test).^2);
        errs(fold) = mse;

    end

    % mean error across all folds
    cv_err = mean(errs);

end