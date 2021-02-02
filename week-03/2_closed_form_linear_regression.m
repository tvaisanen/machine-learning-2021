% Load the data set into matrix A from 'student_debt_data.csv' using the load function
A = load('student_debt_data.csv');

[N,M] = size(A);

% Construct the design matrix X with augmented ones
X = [ones([N,1]),A(:,1)]

% Construct the expected outcome vector y
y = A(:,2)

% Solve the weights using Pseudoinverse
w = pinv(X)*y

x = A(:,1);
model = @(x, w) (w(1) + x' .* w(2:end))';

% Use the model to extrapolate year 2030 debt
y2030 = model([2030],w);

figure

scatter( [A(:,1)], [A(:,2)] )
hold
plot(x, model(x,w))

