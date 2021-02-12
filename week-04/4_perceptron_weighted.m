A = load('3d_classification_data_v2_mbalanced.csv');

X = A(1:2,:)';

y = A(3,:)';

betas = ones( size(y) );

w = trainPerceptronWeighted(X,y,betas);

function w = trainPerceptronWeighted(X,y,betas)

        alpha = 0.1;
        max_iter = 1;

        xs = linspace(1,max_iter+1,max_iter+1)';

        X_aug = [ones([size(X,1),1]), X];

        size(X_aug);

        g = @(w) (1/size(X_aug,2))*sum(betas .* real(log(1 - exp((-1)*y .* X_aug * w'))));
        % g = @(w) (1/size(X_aug,1))*sum(max([zeros([size(X_aug,1),1]), (-1) * y .* X_aug * w']'));

        w0 = randn([1,size(X_aug,2)]);
        
        % this is failing because the gradientDescentAD can not
        % deal with complex numbers.. Can not really figure out
        % from where the complex is coming
        g(w0)
        
        [gw, w, g_history, w_history] = gradientDescentAD(g, w0, alpha, max_iter);

        size(g_history);

        subplot(211)
        plot(xs, g_history)
        subplot(212)
        plot(xs, w_history)

end

