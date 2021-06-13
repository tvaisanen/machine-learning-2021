% Load the data set
A = load('noisy_sin_sample.csv');

% Step size
ALPHA = 0.03;
MAX_ITER = 3000 ;
%theta0 = [      0.2    6.5830    0.152    1.0150 ];
theta0 = [0.1393    5.9830    0.0152    1.0150]

w0 = [theta0(3); theta0(4)];
v = [theta0(1); theta0(2)];

x = linspace(0,1,100);

X0 = A(:,1);
Y0 = A(:,2);

y_m  = [ones(size(X0)), sin(v(1) + v(2) * X0)] * theta0(3:4)';


% scatter(x, y_m);
scatter(X0,Y0);
hold on
plot(x, [ones(size(x')), sin(v(1) + v(2) * x')] * theta0(3:4)','r')
% cost = (1/size(X0,2)) * (sum(model(X0, theta0) - Y0) .^ 2) 
%ft = feature_transform(x, v);
[theta, cost_history, theta_history] = fitSingleOutputRegression( A(:,1), A(:,2), theta0, ALPHA, MAX_ITER  );
cost_history(end)
theta
plot(x, [ones(size(x')), sin(v(1) + v(2) * x')] * theta(3:4)','b')
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

    cost = @(theta) (1/size(X,1)) * sum(([ones(size(X)), sin(theta(1) + theta(2) * X)] * theta(3:4)' - y) .^ 2);

    [gw, theta, cost_history, theta_history] = gradientDescentAD(cost, theta0, alpha, max_iter);

end 

    