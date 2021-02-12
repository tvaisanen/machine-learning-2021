% Load X and y from 'spambase_data.mat'
load('spambase_data.mat');

size(X)
size(y)

% Call your training function
w = trainPerceptronNormalized(X,y)

w'

function w = trainPerceptronNormalized(X,y)

    alpha = 0.1;
    max_iter = 2000;

    xs = linspace(1,max_iter+1,max_iter+1)';

    X_aug = [ones([size(X,1),1]), normalize(X)];

    size(X_aug)
    
    

    g = @(w) (1/size(X_aug,1))*sum(max([zeros([size(X_aug,1),1]), (-1) * y .* X_aug * w']'));

    % w0 = randn([1,size(X_aug,2)]);

    w0 = [
       -0.6146
   -0.5617
    0.2502
    0.2158
   -0.9992
    0.1768
   -0.2997
   -0.1252
    0.8446
    0.4037
    0.4076
    0.8750
    0.0077
    0.5370
    0.6023
    1.1626
    0.6792
    1.3915
    0.1776
    0.3680
    1.7122
    0.0456
    0.1856
    1.1656
    0.3955
    0.6711
    0.3968
    0.1133
   -1.0602
    0.1857
   -1.6838
   -0.0722
    0.5159
    0.0398
    0.5914
    1.6173
    0.5187
   -0.0191
    0.4947
    0.1275
   -1.5025
   -0.4643
    0.7162
   -0.2755
    0.2835
   -1.0622
   -0.0028
    0.3834
   -1.3929
    0.1793
   -0.2719
   -0.9534
    0.3266
   -1.2996
   -1.9486
   -0.9304
   -0.6329
   -0.6965
    ]';
    
    
    [gw, w, g_history, w_history] = gradientDescentAD(g, w0, alpha, max_iter);

    size(g_history)

    subplot(211)
    plot(xs, g_history)
    subplot(212)
    plot(xs, w_history)


    
end