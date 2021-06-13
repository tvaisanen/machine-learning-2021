
% Load the data
A = load('credit_dataset.csv');

% Features are on the rows; upto the second last
X = A(1:end-1,:)';

% Normalize the data for learning
X = normalize(X);

w0 = randn([21,1]);
X_aug = [ones([size(X,1),1]), X];


% Class label is on the last row
y = A(end,:)';

% The set of regularization parameters to test
LAMBDAS = [0 0.01 0.02 .04]';

% Perform l1-regularized softMax training
[W, cost_history] = regularizedFeatureSelection(X,y,LAMBDAS);
W

% Plot the result (not mandatory, but beneficial)
N = size(X,2);          % Number of features
L = length(LAMBDAS);    % Number of lambdas
h = [];
figure

for i = 1:L
    h(i) = subplot(L,1,i);
    bar( 0:N, W(i,:) )
    xticks( 0:N+1 )
    title( sprintf('Weight values at lambda %.2f', LAMBDAS(i)), sprintf('Cost = %.2f', cost_history(i) ) )
end
linkaxes(h)

function [W, cost_history] = regularizedFeatureSelection(X,y,lambdas)

    % Initialize variables
    X_aug = [ones([size(X,1),1]), X];

    % Number of regularization parameters lambda to test
    L = length( lambdas );
    
    w0 = [
        1.1322
        0.7224
       -0.2940
        0.4040
        0.0777
       -0.2504
        0.3671
        0.1768
       -0.3189
        0.1755
        0.1584
       -0.0100
       -0.1857
        0.0951
        0.1644
        0.1468
       -0.1303
        0.0057
       -0.0544
        0.1376
        0.2066
    ]';
    
    W = [];
    cost_history = [];
    % Go through all the lambdas
    for i = 1:L

        % Call the local function
        [w, cost] = trainPerceptronL1(X_aug, y, lambdas(i), w0 )
        
        % Return trained weights in the matrix W
        W = [W;w];
        % Return the training cost in the cost_history
        cost_history(end+1) = cost;
    end
    
end

function [w, cost] = trainPerceptronL1(X,y,lambda, w0)
    
    max_iter = 1000;
    alpha = 0.01;
    
    g = @(w) (1/size(X,1))*sum(log(1 + exp((-1)*y .* X * w'))) + lambda * sum(abs(w));
    
    [cost, w, cost_history, weight_history] = gradientDescentAD(g,w0,alpha,max_iter);

end
