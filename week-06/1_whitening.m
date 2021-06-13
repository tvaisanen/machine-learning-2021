%{
T = [ -3    3
       2   -1
       1    2
      -0.5 -2];
 
Ts = T * T';

P = size(T,1);
N = size(T,2);

mT = mean(T);
T0 = T -mT;

scatter(T(:,1), T(:,2),'or')
hold on
scatter(T0(:,1), T0(:,2),'ob')

[U, S, V] = svd(T0,'econ');

U*S*V'

W = sqrt(P - 1) * V * inv(S);
Tw = T0 * W;
cov(Tw)

%}
% File 'cifar_data.mat' contains matrix A and vector (uint8 data type)
load('cifar_data.mat');

% Convert to double data type for computations
X = double(A);
x = double(a);

P = size(X,1); % Number of samples
N = size(X,2); % Number of features

mX = mean(X);  % Find data center
X0 = X - mX;   % Compute the centered data

[U, S, V] = svd(X0, 'econ');

% Compute whitening transformation W so that Xw = X0 * W
W = sqrt(P - 1) * V * inv(S);

% Compute whitened data matrix Xw so that cov( Xw ) = eye( N )
Xw = X0 * W; 
    
[u,s,v] = svd(x' * x);
% Transform the new sample x into whitened feature space
xw = (x - mean(x)) * W;

% Plot the result (not mandatory, but beneficial)
% Show first MxM images
M = 5;

figure
for i = 1:M^2
    subplot(M,M,i)
    imshow( reshape( X(i,:), 32, 32 )', [0 255] )
end
subtitle('Original data')

range = max(abs(Xw(:)));
figure
for i = 1:M^2
    subplot(M,M,i)
    imshow( reshape( Xw(i,:), 32, 32 )', [-range range] )
end
subtitle('Whitened data')

figure
subplot(121)
imshow( reshape( x, 32, 32 )', [0 255] )
title('Original new sample')
subplot(122)
imshow( reshape( xw, 32, 32 )', [-range range] )
title('Whitened new sample')


