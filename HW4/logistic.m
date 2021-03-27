% Logistic Regression Hw Problem
% CSE 847
% Reuben Lewis, 3/31/21

clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables.

% Load the dataset and initialize the needed arrays
data = load('email_data.txt');
labels = load('email_labels.txt');

[rows, ~] = size(data);
data = [data ones(rows, 1)];

% Split the data into test and training data
trainingData = data(1:2000, :);
trainingLabels = labels(1:2000);

testingData = data(2001:end, :);
testingLabels = labels(2001:end);

% Test the logistic train function with the full set
figure
weights = logistic_train(trainingData, trainingLabels);
predictions = predict(testingData, weights);

% Find the accuracy by checking the different values and seeing if they
% are equal between the test labels and the predictions.
accuracy = getAcc(testingLabels, predictions)

% Create the graph to show the accuracy for each set of training data
acc = zeros(1, 6);
idx = 1;
figure
hold on
for i = [200, 500, 800, 1000, 1500, 2000]
    weights = logistic_train(trainingData(1:i, :), trainingLabels(1:i, :));
    predictions = predict(testingData, weights);
    acc(idx) = getAcc(testingLabels, predictions)
    idx = idx + 1;
end
hold off
figure
plot([200, 500, 800, 1000, 1500, 2000], acc);


function [weights] = logistic_train(data, labels, lr, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
%   data    = n*(d+1) matrix withn samples and d features, where
%             column d+1 is all ones (corresponding to the intercept term)
%   labels  = n*1 vector of class labels (taking values 0 or 1)
%   epsilon = optional argument specifying the convergence
%             criterion - if the change in the absolute difference in
%             predictions, from one iteration to the next, averaged across
%             input features, is less than epsilon, then halt
%             (if unspecified, use a default value of 1e-5)
%   maxiter = optional argument that specifies the maximum number of
%             iterations to execute (useful when debugging in case your
%             code is not converging correctly!)
%             (if unspecified can be set to 1000)
%
% OUTPUT:
%    weights = (d+1)*1 vector of weights where the weights correspond to
%              the columns of "data"
%
    arguments
        data
        labels
        lr = 1e-3
        epsilon = 1e-5
        maxiter = 1000
    end
    [numRows, numCols] = size(data);
    
    % Init the weight vector
    weights = zeros(1, numCols);
    loss = zeros(1, maxiter);
    x = linspace(1, maxiter, maxiter);
    
    for i = 1:maxiter
        % Compute the current prediction, then find the gradient and use it
        % to adjust the weights.
        y_hat = sigmoid(data, weights);
        dL = (y_hat - labels).'*data;       
        weights = weights - lr * dL;
        
        % Calculate the loss here. If we are less than our epsilon, exit.
        % Here, use cross entropy, since our labels are 0 and 1.
        y_hat = sigmoid(data, weights);
        loss(i) = -(labels.'*log(y_hat) + (1 - labels).'*log(1 - y_hat));

        if loss <= epsilon
            break
        end
    end
    
    p = plot(x, loss);
    p.DisplayName = string(numRows);
end

function sig = sigmoid(data, weights)
% Calculate the sigmoid for the data and weights combined. Be sure to do
% this elementwise.
    sig = 1./(1 + exp(-data*weights.'));
end

function [predictions] = predict(data, weights)
% To predict, apply the sigmoid to get our list of potential values. Then,
% if they are greater than 0.5 (our decision function), change the labels
% to 1, otherwise, to 0.
    predictions = sigmoid(data, weights);
    predictions(predictions >= 0.5) = 1;
    predictions(predictions < 0.5) = 0;
end

function [accuracy] = getAcc(test, predictions)
% Find the accuracy by comparing the test labels to the prediction labels
% and the finding the percent accurate.
    [vals, ~] = size(test);
    correctVals = 0;
    for label = 1:vals
       if predictions(label) ==  test(label)
          correctVals = correctVals + 1; 
       end
    end
    accuracy = (correctVals / vals) * 100;
end