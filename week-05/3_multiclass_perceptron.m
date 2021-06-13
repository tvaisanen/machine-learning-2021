% Load the data from the file named '3class_data.csv'
A = load('3class_data.csv');

% Form the feature matrix starting from the first row upto second last
X = A(1:end-1,:)';

% Form the class column vector from the last row in the data
y = A(end,:)';

% Number of classes
C = 3;

% Call the training function
W = trainMultiClassPerceptron(X,y,C)

% This function is the one implemented in the first problem
c = classifyMultiClass( W, X );

accuracy = 100 * sum( c==y ) / length(c)

figure
scatter( X(:,1), X(:,2), 25, y, 'filled' )
hold on
scatter( X(:,1), X(:,2), 60, c )
hold on

axis([0 1 0 1])
xlabel('x_1')
ylabel('x_2')
title( sprintf('Classification accuracy %.2f %%', accuracy ) )



function W = trainMultiClassPerceptron(X,y,C)

    P = size(X,1);
    X = [ones([P,1]), X];
    
    alpha = 0.1;
    max_iter = 100;
    
    g = @cost_perceptron;

    
    w0= [
        5 0 0
        9 1 3
        1 1 0.3
     ] * 0.001;

    
    [g, w, g_history, w_history] = gradientDescentAD(g, reshape(w0,[1,9]), alpha, max_iter);
    
    xs = linspace(1,size(g_history,1), size(g_history,1));
    plot(xs, g_history);
    w
    cost = g_history(end)
    % Return the best weight vector but in matrix form
    W = reshape(w, [3,3]);
    
    % Nested cost function
    function c = cost_perceptron(w)
        w = reshape(w, [3,3])';    
        
        a = max(X*w',[],2);
        b = sum(X.*w(y+1,:),2);
    
        c = 1/P * sum(sum(max(a-b)));

    end

end
