% Load the data set
A = load('breast_cancer_data.csv');

% Form the feature matrix. The first 8 rows in the data set are features.
X = A(1:end-1,:)';

% Form the class label vector. The last row in the data set is the class.
y = A(end,:)';

% Call your training function
w = trainPerceptron(X,y);
w'

% X * randn(1,8)'

% g = @(w) (1/size(X_aug,1))*sum(max([zeros([size(X_aug,1),1]), y .* X_aug * w']'));


function w = trainPerceptron(X,y)

        alpha = 0.0009;
        max_iter = 960;


        xs = linspace(1,max_iter+1,max_iter+1)';

        X_aug = [ones([size(X,1),1]), X];

        size(X_aug);

        g = @(w) (1/size(X_aug,1))*sum(max([zeros([size(X_aug,1),1]), (-1) * y .* X_aug * w']'));

        %w0 = randn([1,size(X_aug,2)]);
        % w0 = [-0.9668 0.0732 0.0102 0.0593 0.0328 0.0270 0.0662 0.0214 0.0517];

        w0 = [0.1360 -0.0064 0.0006 -0.0065 -0.0030  -0.0028 -0.1069 -0.0021 -0.0099];
        [gw, w, g_history, w_history] = gradientDescentAD(g, w0, alpha, max_iter);

        size(g_history)

        subplot(211)
        plot(xs, g_history)
        subplot(212)
        plot(xs, w_history)



end

% If necessary, define local helper functions below
