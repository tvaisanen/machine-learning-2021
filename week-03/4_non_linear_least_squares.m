% Load the data set
A = load('noisy_sin_sample.csv');

% Step size
ALPHA = 0.0003;

% Upper limit of iterations
MAX_ITER = 2000;

% Initial point
% theta0 = [ 0.7    5.3    0.0357    1.0000 ];
theta0 = [ 0.4    5.4    0.00357    1.0000 ];

w0 = [theta0(3); theta0(4)];
v = [theta0(1); theta0(2)];

x = linspace(0,1,100);

y_m  = model(x, theta0);

X0 = A(:,1);
Y0 = A(:,2);

% scatter(x, y_m);
scatter(X0,Y0);
hold on
plot(x, model(x, theta0),'r')
cost = (1/size(X0,2)) * (sum(model(X0, theta0) - Y0) .^ 2) 
ft = feature_transform(x, v);
[theta, cost_history, theta_history] = fitSingleOutputRegression( A(:,1), A(:,2), theta0, ALPHA, MAX_ITER  );
cost_history(end)
theta
plot(x, model(x, theta),'b')
% Plot the result (not mandatory, but beneficial)
figure
subplot(211)
plot( cost_history )
title('Cost history', 'r' )
xlabel('Iteration number')
ylabel('Cost (g(w))')
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')

subplot(212)
plot( theta_history )
title('Theta history', 'r' )
xlabel('Iteration number')
ylabel('Parameter value')



% The main function to do the non-linear fitting
function [theta, cost_history, theta_history] = fitSingleOutputRegression( X, y, theta0, alpha, max_iter )

    % << IMPLEMENT THE FUNCTION BODY! TYPICAL STEPS ARE GIVEN IN COMMENTS BELOW >>

    % Initialize variables
    
    w0 = [theta0(3); theta0(4)];
    v = [theta0(1); theta0(2)];
    cost = @(theta) (1/size(X,2)) * (sum(model(X, theta) - y) .^ 2);

    [gw, theta, cost_history, theta_history] = gradientDescentAD(cost, theta0, alpha, max_iter);
    % Return the best solution, and the histories
    
    % This function computes the least squares cost function
    % NOTE: As a nested function, it can use X and y directly and needs only the parameter vector theta
    %function c = cost(theta)
    %    c = (1/size(X,2)) * (sum(model(X, theta) - y) .^ 2);
    %end

end % End of function fitSingleOutputRegression

% Local helper functions below

% This function transforms the features x non-linearly using the parameters v
function z = feature_transform( x, v )
    z = [1 sin(v(1) + v(2) *x)]';
end

% This function applies the model specified by the parameters theta to the data x
function y = model(x, theta)
    y = theta(3) + sin(theta(1) + theta(2) * x);
end
