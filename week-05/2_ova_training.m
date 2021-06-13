% Load the data set from the file '3class_data.csv'
A = load('3class_data.csv');

% Form the feature matrix starting from the first row upto second last
X = A(1:end-1,:)';

% Form the class column vector from the last row in the data
y = A(end,:)';

% Number of classes
C = 3;

% Call the training function
W = trainOneVsAll( X, y, C )

c = classifyMultiClass( W, X );


[y, c, c == y]

sum(c-1 == y)
size(y, 1)

xs = linspace(0,1,10);
x = [ones(size(xs))', xs'];

% Plot the result (not mandatory, but beneficial)
figure
scatter( X(:,1), X(:,2), 25, y, 'filled' )
hold on
scatter( X(:,1), X(:,2), 60, c )

axis([0 1 0 1])
xlabel('x_1')
ylabel('x_2')
legend('true class', 'predicted class')



function W = trainOneVsAll( X, y, C )

    X_aug = [ones([size(X,1),1]), X];
    lambda = 0.00001;
    
    for i = 1:C
        
        y_p = (y==(i-1)) * 2 - 1;
        [w, cost_history, w_history] = trainPerceptron(X, y_p,lambda);
        
        xs = linspace(1,size(cost_history,1), size(cost_history,1));
        plot(xs, cost_history);
        W(i,:) = w/norm(w(2:end));
        
    end
    
    W = W;
    
end

function c = classifyMultiClass( W, X )

    X_aug = [ones([size(X,1),1]), X];
    
    y_p =(X_aug * W');
 
    [a,b] = max((X_aug * W')');
    c = (b - 1)';
end