% Load the data from 'regression_outliers.csv' using the load function
A = load("regression_outliers.csv");

% Load the weights from 'problem1.mat' using the load function
load('problem1.mat')

% Create 100 evenly spaced grid of points between -2 and 2 (inclusive) for model evaluation and plotting. Create a column vector
x = linspace(-2,2,100)';

P = size(A,2);
y_p = mean(x)

model = @(x, w) (w(1) + x' .* w(2:end))';

% Evaluate the LS model at x, i.e. use w_LS to calculate output at the points in x. Create a column vector of results
y_LS = model(x,w_LS);

% Evaluate the LAD model at x, i.e. use w_LAD to calculate output at the points in x.  Create a column vector of results
y_LAD = model(x,w_LAD);

% Construct the Least Squares cost function
cost_LS = @(w)  sum((model(A(1,:),w) - A(2,:)) .^ 2) / P;

% Construct the Least Absolute Deviations cost function
cost_LAD = @(w) (sum(abs(model(A(1,:),w) - A(2,:)))) / P;

% Compute the LS cost on weights w_LS and w_LAD
cost_LS_wLS =  cost_LS(w_LS) % Evaluate cost_LS at w_LS
cost_LS_wLAD = cost_LS(w_LAD) % Evaluate cost_LS at w_LAD

% Compute the LAD cos%t on weights w_LS and w_LAD
cost_LAD_wLS = cost_LAD(w_LS) % Evaluate cost_LAD at w_LS
cost_LAD_wLAD = cost_LAD(w_LAD) % Evaluate cost_LAD at w_LAD

% Plot the result
figure

scatter( A(1,:), A(2,:) )
hold on
plot( x, y_LS )
hold on
plot( x, y_LAD, '--' )
legend('data','Least Squares', 'Least Absolute Deviations','Location','NorthWest')
axis([-2 2 -5 12 ])
xlabel('x')
ylabel('y')
