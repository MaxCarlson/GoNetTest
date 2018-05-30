clear ; close all; clc

inputSize = 361; % 19x19 go board
layer2Size = 25;
layser3Size = 25; % Scale these up

outputs = 361; % Lets get a probability for each board position

data = load('goData.txt');

y = data(:, end);
X = data(:, 1:end-1);

