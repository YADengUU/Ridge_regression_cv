classdef test_errors < matlab.unittest.TestCase

    methods(Test)
        function testInadequateInput(testCase)
            X=ones(2500,999);
            func=@()ridge_regression(X);
            testCase.verifyError(func,'MyError:InadequateInput');
        end
        
        function testNonNumericInputX(testCase)
            X=['a','b','c';'d','e','f';'1','2','3'];
            y=ones(3,1);
            func=@()ridge_regression(X,y);
            testCase.verifyError(func,'MyError:NonnumericalInput');
        end

        function testNonNumericInputY(testCase)
            X=ones(3,3);
            y=['1';'c';'2'];
            func=@()ridge_regression(X,y);
            testCase.verifyError(func,'MyError:NonnumericalInput');
        end

        function testWrongSize(testCase)
            X=ones(5,5);
            y=ones(4,1);
            func=@()ridge_regression(X,y);
            testCase.verifyError(func,'MyError:UnmatchingSize');
        end

        function testNonNumericLambda(testCase)
            X=ones(5,5);
            y=ones(5,1);
            lambdas=['1','0.1'];
            func=@()ridge_regression(X,y,lambdas);
            testCase.verifyError(func,'MyError:NonnumericalLambda');
        end

        function testSmallTrainRatio(testCase)
            X=ones(5,5);
            y=ones(5,1);
            func=@()ridge_regression(X,y,exp(-5:5),0.5);
            testCase.verifyError(func,'MyError:SmallTraining');
        end

        function testHugeTrainRatio(testCase)
            X=ones(5,5);
            y=ones(5,1);
            func=@()ridge_regression(X,y,exp(-5:5),2);
            testCase.verifyError(func,'MyError:HugeTraining');
        end

        function testNonNumericTrainRatio(testCase)
            X=ones(5,5);
            y=ones(5,1);
            func=@()ridge_regression(X,y,exp(-5:5),'0.75');
            testCase.verifyError(func,'MyError:NonnumericalTraining');
        end

    end
        
end