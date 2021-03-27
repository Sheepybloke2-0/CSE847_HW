% Ridge Regression Hw Problem
% CSE 847
% Reuben Lewis, 3/17/21

clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables.

% Load the dataset and initialize the needed arrays
data = load('diabetes.mat');

% Iterate through the different lamdas, train based on the lamda, then
% figure out how we've done with the training.
figure
hold on
for lamda = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

    [weights, b, error, x_mse, y_mse] = ridgeRegular(data.x_train, data.y_train, lamda, 15000, 0.001);

    prediction = predict(data.x_test, weights, b);
    mean((data.y_test - prediction).^2)
    error
    p = plot(x_mse, y_mse);
    p.DisplayName = string(lamda);

end
hold off

totalErrors = zeros(7,1);
totalTrainErrors = zeros(7,1);
index = 1;
for lamda = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    cvError = zeros(5,1);
    trainError = zeros(5,1);
    
    % Split the data into 5 different partitions.
    splitPoints = [1, 49, 97, 145, 193, 242];
    for i = 1:5
        x_train = data.x_train;
        x_train(splitPoints(i):splitPoints(i+1), :) = [];
        x_test = data.x_train(splitPoints(i):splitPoints(i+1), :);
        y_train = data.y_train;
        y_train(splitPoints(i):splitPoints(i+1), :) = [];
        y_test = data.y_train(splitPoints(i):splitPoints(i+1), :);
        
        [weights, b, error, ~, ~] = ridgeRegular(x_train, y_train, lamda, 15000, 0.001);
        
        prediction = predict(x_test, weights, b);
        cvError(i) = mean((y_test - prediction).^2);
        trainError(i) = error;
    end
    
    totalErrors(index) = 0.2*sum(cvError)
    totalTrainErrors(index) = 0.2*sum(trainError)
    index = index + 1;

end

figure
hold on
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
lamda = logspace(-5, 1, 7); % [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10];
loglog(lamda, totalErrors, "*")
loglog(lamda, totalTrainErrors,"*")
legend("Total Error", "Train Error");
hold off


function [weights, b, error, x_mse, y_mse] = ridgeRegular(x, y, lamda, iter, lr)
    [~, numCols] = size(x);
    
    % Init the weight vector
    weights = zeros(numCols, 1);
    b = 0;
    y_mse = zeros(1, iter);
    x_mse = linspace(1, iter, iter);
    
    for i = 1:iter
        % Calculate the specific gradient here.
        pred = predict(x, weights, b);
        dW = -2 * x.'*(y - pred) + 2*lamda*weights ;
        dB = -2 * sum(y - pred);
        
        % Update the weights based on the learning rate
        weights = weights - lr * dW;
        b = b - lr * dB;
        
        prediction = predict(x, weights, b);
        y_mse(i) = mean((y - prediction).^2);
    end
    
    error = mean((y - prediction).^2);
end

function predictions = predict(x, weights, b)
    predictions = x*weights + b;   
end
