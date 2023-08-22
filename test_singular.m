% test class for extreme input

%  OLS cannot cope with singular input matrix
classdef test_singular < matlab.unittest.TestCase
    methods(Test)
        function testOutputValues(testCase)
            X=zeros(500,199);
            num_params=199;
            low_bound=-20;
            high_bound=20;
            beta = low_bound + (high_bound-low_bound)*randn(num_params+1,1);
            epsilons = 0.5 * randn(500,1);
            y = [ones(500,1),X]*beta+epsilons;
            lambdas = exp(-9:5);
            [b,b_ols,~,~,~,~,~,~] = ridge_regression(X,y,lambdas);
            testCase.verifyEqual(b(2:end),zeros(199,1),'Ridge: Estimated coefficient values (b) except the constant term should be zero.');
            testCase.verifyTrue(all(isnan(b_ols(2:end))),'OLS: Singular matrix is not invertible thus unable to compute the coefficients.');

        end
    end
end