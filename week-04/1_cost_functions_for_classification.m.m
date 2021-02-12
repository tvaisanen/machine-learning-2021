% Load the data from the file '2d_classification_data_v1_entropy.csv' using the load function
A = load('2d_classification_data_v1_entropy.csv');

[~, P] = size(A);

x = A(1,:)'; 
y = A(2,:)';

x_aug = @(x) [ones([1,size(x,2)]); x];

% Define the step function, for t=0 return 1. Return value in the same data type as t or a double
step = @takeStep;    
x_size = size(x)
model = @(x, w) w(1) + x * w(2:end);
% model = @(x, w) x_aug(x)' * w;
% model = @(x, w) dot([ones([size(x,2),1]), x'], w);
% model = @(x, w) (w(1,:) + x' .* w(2:end))';

% Define the step model function using the step function
model_step = @(x,w) step(model(x,w));  

% Define the step cost function using the model_step
% this is linear regression with model_step
cost_step_LS = @(w) ((1/P) * sum(model_step(x,w) - y));     

% Define the sigmoid function
sigmoid = @(x) arrayfun(@(x) (1/(1 + exp(-x))), x);

% Define the Logistic regression model using sigmoid function
model_logit = @(x,w) sigmoid(model(x,w));  

% Define the Logistic regression Least Squares cost function
cost_logit_LS = @(w) ((1/P) * sum(model_logit(x,w) - y)); % IMPLEMENT THE ANONONYMOUS FUNCTION

% Define the Logistic regression Cross-Entropy cost function
cost_logit_CE = @(w) ((1/P) * y .* log(model_logit(x, w)) + 
                      (1 - y) .* log(1 - model_logit(x, w))));  
% IMPLEMENT THE ANONONYMOUS FUNCTION

% 

% Plot the result (not mandatory, but beneficial)

% Initialíze a regular rectangular 2D grid of points to evaluate the cost functions on
% Create 100 equidistantly spaced samples from -20 to +20 (limits inclusive)
w_range = linspace(-20,20,100);
% Use 'w_range' to construct a 2D mesh using the meshgrid function
[XX,YY] = meshgrid( w_range );

% (Almost) optimally picked weights
w_opt = [ -10.38 10.03 ];
% (Clearly) unoptimally picked weights
w_unopt = [ -1.5 2 ];
% Range of x-values from -1 to 6; to evaluate the model outputs
x_range = linspace(-1,6,100)';


size(x_range)
size(w_opt)
XR = linspace(-1,5,10)';


% Plot data together with both the optimal and unoptimal step and logit models 
figure
scatter( A(1,:), A(2,:) )
axis([-1 5 -1 2])
hold on
plot( x_range, model_step( x_range, w_opt ) )
plot( x_range, model_logit( x_range, w_opt ) )
plot( x_range, model_step( x_range, w_unopt ), '--' )
plot( x_range, model_logit( x_range, w_unopt ), '--' )
xlabel('x')
legend('Data', 'Step opt.', 'Logit opt.', 'Step unopt.', 'Logit unopt.' )
title('Classification regression problem')

% Plot the cost function surfaces in 3D
%cost_functions = { cost_step_LS, cost_logit_LS, cost_logit_CE };
%cost_function_names = { 'Step LS cost', 'Logit LS cost', 'Logit CE cost' };
%M = length( cost_functions );
%v = [-152 42];
%figure
%for i = 1:M
%    subplot(M,1,i)
%    surface(XX,YY, arrayfun( @(x,y) cost_functions{i}( [x y] ), XX, YY ) )
%    view(v)
%    xlabel('w_0')
%    ylabel('w_1')
%	zlabel('cost')
%    title( cost_function_names{i} )
%end


function s = takeStep(t)
   t( t >= 0 ) = 1;
   t( t < 0 ) = 0;
   s = t;
end
    