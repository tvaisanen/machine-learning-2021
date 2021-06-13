% Load the data from the file named 'ellipse_2class_data.csv'
A = load('ellipse_2class_data.csv');

% Form the feature matrix from the first and second row of the data
X = A(1:2,:)';

% Form the class column vector from the last row in the data
y = A(3,:)';





% Call your training function
% [1,3] = size(w)


w = fitNonlinearSoftmax(X,y)


function theta = fitNonlinearSoftmax( X, y )

    w0 = [14.9255  -21.4804  -44.0092];
    
    [P,N] = size(X);
    
    alpha = 10;
    max_iter = 100;
    
    % MODEL
    x_aug = @(x)[ones([P,1]), x(:,1), x(:,2)];
    Xa = x_aug(X);
    
    model = @(x,w) (x .^ 2) * w';
    
    g = @(w) (1/P)*sum(log(1 + exp((-1) * y .* model(Xa,w))));
    

    [gw, theta, g_history, theta_history] = gradientDescentAD(g,w0,alpha,max_iter);
    
    xs = linspace(1,max_iter+1,max_iter+1)';
    
    cost = g_history(end)
    
    subplot(211)
    plot(xs, g_history)
    subplot(212)
    plot(xs, theta_history)

end

