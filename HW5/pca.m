% PCA Hw Problem
% CSE 847
% Reuben Lewis, 4/21/21

clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables.

load USPS.mat
for i = 1:2
    % Display our reference
    num = randi(3000);
    A1 = reshape(A(num, :), 16, 16);
    figure
    imshow(A1');
    
    for i = [10, 50, 100, 200]
        returned = pca_hw(A, i);
        A1_pca = reshape(returned(num,:), 16, 16);
        error = recon_error(A, returned)
        figure
        imshow(A1_pca');
    end
end

function [output] = pca_hw(data, pcs)
    % Get the covariance of the data matrix
    covar = cov(data);
    % Get the eigenvalues for that covariance matrix
    [evec, evals] = eig(covar);
    
    % Sort the eigenvectors from largest to smallest
    [~,idx] = sort(diag(evals), 'descend');
    evec = evec(:,idx);
    
    % Grab the top PCs of the matrix
    evec = evec(:, 1:pcs);
    
    % Transpose the data into the reduced dimensions and then out to the
    % original data again
    finalData = evec.' * data.';
    output =  evec * finalData;
    output = output.';
end

function [error] = recon_error(orignal, pca)
    error = sum((orignal - pca).^2, 'all');
end