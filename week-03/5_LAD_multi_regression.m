

% Load the data set from the file 'student_debt_data.csv'
A = load('linear_2output_regression.csv');

% Input variables as a matrix with samples on rows
X = A(1:2,:)';

% Output variables as a matrix with samples on rows
Y = A(3:end,:)';

% Call your fitting function
W = fitMultipleOutputRegression( X, Y )


% Plot the result (not mandatory, but beneficial)

% Initialize a regular mesh to plot the fitted plane
xx = 0:0.1:1;           % Span of mesh data
[XX,YY] = meshgrid(xx); % Create the mesh

% Number of output variables 
C = size(Y,2);

% Make the figures
figure
for i = 1:C
    
    subplot(C,1,i)

    scatter3( X(:,1), X(:,2), Y(:,i), 'filled', 'k' )
    view(25,25) 
    xlabel('x_1')
    ylabel('x_2')
    zlabel(sprintf('y_%d', i))
    title(sprintf('Plot of output %d samples and the fitted plane', i))
    
    hold on
    
    ZZ = arrayfun( @(x,y) [1 x y] * W(:,i) , XX, YY );
    surf(XX,YY,ZZ)
    alpha(0.5)
    
end


function W = fitMultipleOutputRegression( X, Y )

    [P,N] = size(X);

    X_aug = [ones([P,1]), X];
    
    alpha = 0.01;
    max_iter = 2000;

    C = 2;
    
    g = @gs;

   
    w0= [
        1.0144    0.5932
        0.1456   -0.0309
       -0.2745   -1.0357
    ];
      
    xs = linspace(1,max_iter+1,max_iter+1)';

    for i = 1:C

        [gw, w, g_history, w_history] = gradientDescentAD(g, w0, alpha, max_iter );
        
        gw
        % Store the result in the corresponding column of the output weight matrix
        w
        W(:,i) = w(:,2);
        subplot(2,1,i);
        plot(xs, g_history);
        
    end
    
    plot(xs, g_history)
    
    function gw = gs(w)
       gw = (1/P)*sum(sum(abs(X_aug * w - Y(:,i))));
    end
  
    
end

