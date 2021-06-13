A = load('3d_classification_data_v2_mbalanced.csv');

X = A(1:2,:)';

y = A(3,:)';

betas = ones( size(y) );
% betas = [ones([5,1]);y(6:end)*(-5)]

w = trainPerceptronWeighted(X,y,betas);


function w = trainPerceptronWeighted(X,y,betas)

        % alpha = 2;
        % max_iter = 2000;
        % w0 = [  -17.7330   -10.1939   19.5220]
        %
        % -> 1.5802
        
        % alpha = 1.5;
        % max_iter = 3000;
        % w0 = [-17.7366   10.1960   19.5262]
        %
        % 1.5802
        % 
        % 0.2767; and the reference cost function value is 0.2036
        % 0.3898; and the reference cost function value is 0.2697
        
        alpha = 1.3;
        max_iter = 3000;
        w0 = [-17.7366   10.1960   19.5262]
        %w0 = [-0.9   10.1988   19.5324];
        
        %
        % 1.5802
        % 
        % 0.2767; and the reference cost function value is 0.2036
        % 0.3898; and the reference cost function value is 0.2697
        
        xs = linspace(1,max_iter+1,max_iter+1)';

        X_aug = [ones([size(X,1),1]), X];

        g = @(w) (1/size(X_aug,1))*sum(betas .* log(1 + exp((-1)*y .* X_aug * w')));

        [gw, w, g_history, w_history] = gradientDescentAD(g, w0, alpha, max_iter);

        g(w)
        w

        subplot(211)
        plot(xs, g_history)
        subplot(212)
        plot(xs, w_history)

end

