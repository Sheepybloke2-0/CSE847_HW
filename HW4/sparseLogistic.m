% Sparse Logistic Regression Hw Problem
% CSE 847
% Reuben Lewis, 3/31/21

clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables.

oldpath = path;
path(oldpath, 'SLEP-master\SLEP\functions\L1\L1R')
path(oldpath, 'SLEP-master\SLEP\opts')
                     % add the functions in the folder SLEP to the path

% Load the dataset and initialize the needed arrays
data = load('ad_data.mat');

% par  = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99];
par = linspace(0.001, 0.99999, 1000);
aucArr = zeros(size(par));
featArr = zeros(size(par));
idx = 1;

figure
hold on
for rho = par

    [w, c] = logisticl1train(data.X_train, data.y_train, rho);

    predictions = predict(data.X_test, w, c);

    acc = getAcc(data.y_test, predictions);
    
    sigs = sigmoid(data.X_test, w, c);
    [x, y, t, auc] = perfcurve(data.y_test, sigs, 1);
    
    aucArr(idx) = auc
    
    count = 0;
    for i = size(w)
       if w(i) ~= 0
          count = count + 1; 
       end
    end
    featArr(idx) = count;
    idx = idx + 1;
    
    p = plot(x, y);
    p.DisplayName = string(rho);
    rho, auc, acc

end
hold off

figure
plot(par, aucArr)
figure
plot(par, featArr)

function [w, c] = logisticl1train(data, labels, par)
% OUTPUT w is equivalent to the first d dimension of weights in logistictrain
%        c is the bias term, equivalent to the last dimension in weights in logistictrain.
% Specify the options (use without modification).
    opts.rFlag = 1;  % range of par within [0, 1].
    opts.tol = 1e-6; % optimization precision
    opts.tFlag = 4;  % termination options.
    opts.maxIter = 5000; % maximum iterations.

    [w, c] = LogisticR(data, labels, par, opts);
    
end

function sig = sigmoid(data, weights, bias)
% Calculate the sigmoid for the data and weights combined. Be sure to do
% this elementwise.
    sig = 1./(1 + exp(-data*weights + bias));
end

function [predictions] = predict(data, weights, bias)
% To predict, apply the sigmoid to get our list of potential values. Then,
% if they are greater than 0.5 (our decision function), change the labels
% to 1, otherwise, to 0.
    predictions = sigmoid(data, weights, bias);
    predictions(predictions >= 0.5) = 1;
    predictions(predictions < 0.5) = -1;
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