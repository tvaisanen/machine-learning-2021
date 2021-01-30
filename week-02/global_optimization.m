
% Handle to the function to be minimized
f = @(x) (x.^2 + 0.2);

% For uniform sampling of f(x), create an evenly spaced sampling grid as the row vector 'x1' in the span from x=-1.0 to x=1.0 with 100 sample points.
x1 = linspace(-1,1,100);


% For random sampling of f(x), create a randomly chosen sampling row vector 'x2' in the span from x=-1.0 to x=1.0 with 100 sample points. Pick the points from the continuous uniform random distribution.
x2 = unifrnd(-1,1,[1,100]);


% Sample the function at x1 into 'f_val_x1'
f_val_x1 = f(x1)

% Sample the function at the points in x2 into vector 'f_val_x2'
f_val_x2 = f(x2)

% Find the minimum value of f based on uniform samples in f_val_x1 as variable 'f_min_x1'
f_min_x1 = min(f_val_x1)

% Find the minimum value of f based on random samples in f_val_x2 as variable 'f_min_x2'
f_min_x2 = min(f_val_x2)

% Plot the result (not mandatory, but beneficial)
figure
cats = categorical({'True', 'Uniform sampling','Random sampling'});
cats = reordercats( cats,{'True', 'Uniform sampling','Random sampling'});
range = max([f_min_x1 f_min_x2])-0.2;
bar(cats, [0.2 f_min_x1 f_min_x2])
ylim([max([0.19 0.2-range]) 0.2+range*1.1])
title('Comparison of the results')
xlabel('Method used')
ylabel('Minimum value')