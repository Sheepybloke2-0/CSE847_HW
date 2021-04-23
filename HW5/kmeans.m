% K Means Hw Problem
% CSE 847
% Reuben Lewis, 4/21/21

clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables.

% Generate some data. Create 3 seperate distributions and combine them
t = 2*pi*randn(100, 1);
r = 8*randn(100, 1);
x1 = 50 + r.*cos(t); % Use only x1 and y1 if you want to visualize in 2D
y1 = 30 + r.*sin(t);
z1 = 50 + r.*cos(t);
i1 = 30 + r.*sin(t);
j1 = 50 + r.*cos(t);
k1 = 30 + r.*sin(t);

t = 2*pi*randn(100, 1);
r = 3*randn(100, 1);
x2 = 30 + r.*cos(t);
y2 = 50 + r.*sin(t);
z2 = 30 + r.*cos(t);
i2 = 50 + r.*sin(t);
j2 = 30 + r.*cos(t);
k2 = 50 + r.*sin(t);

t = 2*pi*randn(100, 1);
r = 5*randn(100, 1);
x3 = 25 + r.*cos(t);
y3 = 20 + r.*sin(t);
z3 = 25 + r.*cos(t);
i3 = 20 + r.*sin(t);
j3 = 25 + r.*cos(t);
k3 = 20 + r.*sin(t);

% For visualizing in 2D
% figure
% hold on
% scatter(x1, y1, 'red');
% scatter(x2, y2, 'blue');
% scatter(x3, y3, 'green');
% hold off  

% Create the final dataset
data = [x1, y1, z1, i1, j1, k1; x2, y2, z2, i2, j2, k2; x3, y3, z3, i3, j3, k3];
data = abs(data);

% Run the alternating method and give the SSE
[centers, memberships] = kmeans_alternating(data, 3, 1000);
error_k = sse_k(data, centers, memberships, 3)

% Run the spectral method, which returns SSE from inside
[centers_spc, memberships_spc] = spectral_clustering(data, 3, 4, 1000);

% For visualizing in 2D
% figure
% hold on
% data_k = [data, memberships];
% scatter(centers(:, 1), centers(:, 2), 48,'k', 'x');
% scatter(data_k(data_k(:,3)==1,1), data_k(data_k(:,3)==1,2), 'r');
% scatter(data_k(data_k(:,3)==2,1), data_k(data_k(:,3)==2,2), 'b');
% scatter(data_k(data_k(:,3)==3,1), data_k(data_k(:,3)==3,2), 'g');
% hold off
% 
% figure
% hold on
% data_s = [data, memberships_spc];
% scatter(centers_spc(:, 1), centers_spc(:, 2), 48,'k', 'x');
% scatter(data_s(data_s(:,3)==1,1), data_s(data_s(:,3)==1,2), 'r');
% scatter(data_s(data_s(:,3)==2,1), data_s(data_s(:,3)==2,2), 'b');
% scatter(data_s(data_s(:,3)==3,1), data_s(data_s(:,3)==3,2), 'g');
% hold off

function [centers, memberships] = spectral_clustering(data, numCenters, k, maxiter)
    arguments
        data
        numCenters
        k
        maxiter = 100
    end
    
    % Create the data matrix X.T * X
    matrix = data * data.';
    % Get the Eigenvectors that correspond to each point
    [vec, val] = eig(matrix);
    
    % Extract k elements
    vec = vec(:, 1:k);
    
    % Run the k means to get the potential centroids of each and then get
    % the error.
    [centers, memberships] = kmeans_alternating(vec, numCenters, maxiter);
    error_s = sse_k(vec, centers, memberships, numCenters)
end

function [centers, memberships] = kmeans_alternating(data, numCenters, maxiter)
    arguments
        data
        numCenters
        maxiter = 100
    end
    [~, numCol] = size(data);
    
    mx = max(data);
    mn = min(data);
    
    % Randomly assign locations for the centers in the range
    centers = (mx - mn).*rand(numCenters, numCol) + mn;
    
    for iter = 1:maxiter
        % Get the potential memberships for each piece of data
        memberships = getMemberships(data, centers, numCenters);
        % Update the centers based on the members of each
        centers = getCenters(data, memberships, numCenters);
    end
end

function [memberships] = getMemberships(data, centers, numCenters)
    [numRows, ~] = size(data);    
    dist = zeros(numRows,numCenters);
    for row = 1:numCenters
        for idx = 1:numRows
            % Calculate the distance from that data point to the specific
            % row and hold it in the distance matrix
            dist(idx,row) = sum((data(idx,:) - centers(row, :)).^2);
        end
    end
    
    memberships = zeros(numRows,1);
    
    for row = 1:numRows
        % Get the min distance to each center, and then store that index,
        % which corresponds to the enum of the center.
        [val, idx] = min(dist(row,:));
        memberships(row) = idx;
    end
end

function [centers] = getCenters(data, memberships, numCenters)
    centroidCnt = zeros(1,numCenters);
    [numRow, numCol] = size(data);
    centers = zeros(numCenters, numCol);
    for row = 1:numRow
       idx = memberships(row);
       % Keep track of the number of values per center.
       centroidCnt(idx) = centroidCnt(idx)+ 1;
       % Add the value to the center's running total
       centers(idx, :) = centers(idx, :) + data(idx, :);        
    end
    
    for cent = 1:numCenters
        % If there is no values assigned to the centroid, reassign it
        if centroidCnt(cent) == 0
            mx = max(data);
            mn = min(data);
            centers(cent, :) = (mx - mn).*rand(1, numCol) + mn;
        else
            % Take the average and figure out wher to move the centroids
            centers(cent, :) = centers(cent,:)./centroidCnt(cent);
        end
    end
end

function [error] = sse_k(data, centers, memberships, numClusters)
    data_c = [data, memberships];
    error = 0;
    for c = 1:numClusters
       % Grab the members of the centroid
       members = data_c(data_c(:,end)==c, :);
       err = 0;
       [numRow, ~] = size(members);
       for row = 1:numRow
           % Add to the error for this cluster
           err = err + sum((centers(c, :) - members(row, 1:end-1)).^2);
       end
       error = error + err;
    end

end
