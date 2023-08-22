function [logL] = compute_loglikelihood(y_pred,y_test)
    residuals = y_pred-y_test;
    sig = std(residuals); % residual standard deviation

    logL = 0;
    for i = 1:length(y_pred)
        logL = logL+(-0.5*log(2*pi)-0.5*log(sig^2)-(1/(2*sig^2))*(y_test(i)-y_pred(i))^2);
    end


end