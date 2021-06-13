A = load('3class_data.csv');

% Form the feature matrix starting from the first row upto second last
X = A(1:end-1,:)';

% Form the class column vector from the last row in the data
y = A(end,:)';

% Number of classes
C = 3;

% Trained weights for the 3 One-versus-all classifiers
W = [ -0.0740   -0.7898    0.6134;
      -0.7304    0.9433    0.3319;
       0.4118   -0.0995   -0.9950 ];

% Predict the classes 
c = classifyMultiClass( W, X );

% Plot the result (not mandatory, but beneficial)
figure
scatter( X(:,1), X(:,2), 25, y, 'filled' )
hold on
scatter( X(:,1), X(:,2), 60, c )
axis([0 1 0 1])
xlabel('x_1')
ylabel('x_2')
legend('true class', 'predicted class')



function c = classifyMultiClass( W, X )

    X_aug = [ones([size(X,1),1]), X];
    
    y_p =(X_aug * W')
 
    [a,b] = max((X_aug * W')');
    c = (b - 1)' 
end