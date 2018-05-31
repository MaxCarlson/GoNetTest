clear ; close all; clc

fprintf('Loading data...');
data = load('goData.txt');

y = data(:, end);
X = data(:, 1:end-1);

[m, n] = size(X);

outputSize = inputSize = n; % Go board size
hiddenSize = 55; % Scale these up

% Network weights
% TODO: Add a few more hidden layers
Theta1 = randInitializeWeights(hiddenSize, inputSize); 
Theta2 = randInitializeWeights(outputSize, hiddenSize);

% unroll net parameters
netWeights = [Theta1(:); Theta2(:)];

% Modify y to fit the backprop algo
Y = zeros(size(y, 1), outputSize);
for i = 1:m
  Y(i, :) = zeros(1, outputSize); 
  Y(i, y(i)) = 1;
endfor

lambda = 1;



costFunc = @(p)costFunction(p, inputSize, hiddenSize, ...
                            outputSize, X, Y, lambda);
                            
options = optimset('MaxIter', 105);
[nn_params, cost] = fmincg(costFunc, netWeights, options);

Theta1 = reshape(netWeights(1:hiddenSize * (inputSize + 1)), ...
                 hiddenSize, (inputSize + 1));

Theta2 = reshape(netWeights((1 + hiddenSize * (inputSize + 1)):end), ...
                 outputSize, (hiddenSize + 1));
                 
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);