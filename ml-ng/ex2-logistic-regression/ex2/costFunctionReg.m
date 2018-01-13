function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% h = g(z) z = X * theta
% g(z) = sigmoid(z)
% the first step is to compute J and grad exactly the same as that in costFunction.m
h = sigmoid(X * theta)
J = (-y' * log(h) - (1-y)' * log(1-h)) / m
grad = X' * (h - y) / m


% do not penalize theta zero
theta_reg = theta
theta_reg(1) = 0 
% do regularization
J = J + (lambda / (2 * m)) * sum(theta_reg .^ 2)
grad = grad + (lambda / m) * theta_reg;

% =============================================================

end
