% Load the data set from the file 'student_debt_data.csv'
A = load('student_debt_data.csv');

x = A(:,1);
y = A(:,2);

model = @(x, w) (w(:,1) + x' .* w(:,2:end))';

[P,M] = size(x);

% Construct the cost function
g = @(w) (1/P) * sum((model(x, w) - y) .^ 2);


% Set the step size
ALPHA = 0.00000002;


% Set the upper limit of iterations
MAX_ITER = 500;
w0 = [-144.8 0.07245]

% Set the starting point of iteration
% abs( gw - g_min ) > 0.3
% w0 = [10 10];
%g(w0)

% Solve the weights using GD with AD
[gw, w, g_history, w_history] = gradientDescentAD( g, w0, ALPHA, MAX_ITER );

gw
w
a = g(w0)
b = g(w)
total = abs(a - b)

scatter(x,y)
hold on
plot(x, model(x, w0))
hold on
plot(x, model(x, w), '--')


% Plot the result (not mandatory, but beneficial)
%figure
%subplot(211)
%plot( g_history )
%title('Cost history', 'r' )
%xlabel('Iteration number')
%ylabel('Cost (g(w))')
%set(gca, 'XScale', 'log')
%set(gca, 'YScale', 'log')

%subplot(212)
%f = @(x,y) g([x y]); 
%fcontour(@(x,y) arrayfun(f,x,y), [-1000 1000 -100 100])
%hold on
%plot( w_history(:,1), w_history(:,2), 'r' )
%xlabel('w_1')
%ylabel('w_2')
%title('Cost contour and weight history')


function [gw, w, g_history, w_history] = gradientDescentAD1( g, w0, alpha, max_iter )

    w = w0;
    gw = g(w);


    if nargout > 2
        g_history = [gw];
        w_history = [w];
    end

   h = @(x) wrapper(g,x);

   for i = 1:max_iter

    [fval, gradval] = dlfeval(h,dlarray(w));

    w = w - alpha*gradval;

    [fval, gradval] = dlfeval(h,dlarray(w));

    if nargout > 2
       g_history = [g_history; extractdata(fval)];
       w_history = [w_history; extractdata(w)];
    end

    end

    function [fval, gradval] = wrapper(f,x)
        fval = f(x);
        gradval = dlgradient(fval,x);
    end

   [fval, ~] = dlfeval(h,dlarray(w));

    w = extractdata(w);
    gw = extractdata(fval);

end
