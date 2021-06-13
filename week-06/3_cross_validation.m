% Load the data
load('problem4.mat')

% Set up a range of lambda values
LAMBDAS = 10.^(-3:.25:-1.25);

% Find the best model across LAMBDAS
[ W, best_lambda_ind ] = regularizedL1CrossValidation( y_noisy, y_validate, LAMBDAS );


% Plot the result (not mandatory, but beneficial)
figure

subplot(211)
plot( x, y_true )
hold on;
plot( x, y_noisy, 'x' )
plot( x, y_validate, 'd')
plot( x, W' )
title('All the models')

subplot(212)
plot( x, y_true )
hold on;
plot( x, y_noisy, 'x' )
plot( x, y_validate, 'd')
plot( x, W(best_lambda_ind,:)', 'LineWidth', 3 )
title('The selected model')



function [ W, best_lambda_ind ] = regularizedL1CrossValidation( y_train, y_validate, lambdas )

    % Initialize variables

    % Go through all the lambdas, using parfor for speed instead of for
    parfor i = 1:L
        % Train the model with ith value in lambdas and store weights into correct row of W
        % Call W(...) = regularizedL1Fitting( ... );
    end

    % Determine the best crossvalidation cost and select that model:
    % Compare each model to y_validate in LS cost sense and select least value one
    % Return the index of the best lambda in lambdas as best_lambda_ind
    
end

function [w, cost] = regularizedL1Fitting(y_train, lambda)
    
    % Use output value y as a starting point
    w0 = y_train;

    % Initialize other parameters
    
    % Step size
    ALPHA = % Determine suitable value for fast convergence
    
    % Upper limit on iterations
    MAX_ITER = % Determine suitable value, keep it low enough
    
    % Perform gradient descent
    % Call [...] = gradientDescentAD( ... );
    
    % Pick the best value from the history
        
    % The cost function
    function c = costfun(w)
        
        % LS cost regularized with the L1 norm of Delta^2 w
        c = % Compute
        
    end

end
