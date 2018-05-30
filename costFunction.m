function[cost, grad] = costFunction(nnWeights, ...
                                    inputSize, ...
                                    hiddenSize, ...
                                    outputSize, ...
                                    X, y, lambda)
           

Theta1 = reshape(nnWeights(1:hiddenSize * (inputSize + 1)), ...
                 hiddenSize, (inputSize + 1));

Theta2 = reshape(nnWeights((1 + hiddenSize * (inputSize + 1)):end), ...
                 outputSize, (hiddenSize + 1));
m = size(X, 1);

% Forward prop
% Add X0
a1 = [ones(m, 1), X];

z2 = (a1 * Theta1');
a2 = sigmoid(z2);

% Add a20
a2 = [ones(size(a2, 1), 1), a2];

z3 = (a2 * Theta2');
a3 = sigmoid(z3);

% Cost function
cost = -(1/m) * sum(sum(y .* log(a3) + (1 - y) .* log(1 - a3)));

% This could easily be generalized for any # of layers with a loop 
% and vectors of theta matrix's
cost += (lambda / (2 * m)) * (sum(sum(Theta2(:, 2:end) .^ 2))...
  + sum(sum(Theta1(:, 2:end) .^ 2)));
  
% Backpropagation

% Output node delta
d3 = a3 - y;

% Hidden layer delta
d2 = (Theta2(:, 2:end))' * d3' .* sigmoidGradient(z2)';

% look here for errors!
Delta1 = d2 * a1;
Delta2 = d3' * a2;

% Gradients for each theta
Theta1_grad = (1/m)*Delta1;
Theta2_grad = (1/m)*Delta2;


grad = [Theta1_grad(:) ; Theta2_grad(:)];  
endfunction