function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%h = X * theta
% J = sum((h - y) .^ 2)/(2*m)
predictions = X * theta;              % predictions of hypothesis on examples
sqrErrors   = (predictions - y) .^ 2; % squared errors
%sqrErrors = (X * theta - y)' * (X * theta - y)

J = sum(sqrErrors) / (2 * m);

% =========================================================================

end
