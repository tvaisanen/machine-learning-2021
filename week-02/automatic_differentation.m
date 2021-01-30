



% The quadratic form matrix of the problem
C = [0.1 0; 0 12];

% The function to be minimized. Written this way, the function can be applied to a collection of points on the rows of the matrix w.
g = @(w) dot(w, w*C, 2);

% Starting point for the descent
x0 = [10 1];

% Step size
ALPHA = 0.5e-1;

% Upper limit of iterations
MAX_ITER = 100;

% Call the gradientDescent function
[g_min, w_min, g_history, w_history] = gradientDescentAD( g, x0, ALPHA, MAX_ITER );

% Plot the result (not mandatory, but beneficial)
f = @(x,y)[x y]*C*[x;y]; 
figure
subplot(311)
plot( g_history )
title('Cost history', 'r' )
xlabel('Iteration number')
ylabel('Cost (g(w))')
subplot(312)
fcontour(@(x,y) arrayfun(f,x,y), [9 10.1 -0.3 0.5])
hold on
plot( w_history(:,1), w_history(:,2), 'r' )
axis([9 10.1 -0.3 0.5])
xlabel('w_1')
ylabel('w_2')
title('Weight history (zoomed)')
subplot(313)
fcontour(@(x,y) arrayfun(f,x,y), [-1 10.5 -1.2 1.2])
hold on
plot( w_history(:,1), w_history(:,2), 'r' )
axis([-1 10.5 -1.2 1.2])
xlabel('w_1')
ylabel('w_2')
title('Weight history')



function [gw, w, g_history, w_history] = gradientDescentAD( g, w0, alpha, max_iter )

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