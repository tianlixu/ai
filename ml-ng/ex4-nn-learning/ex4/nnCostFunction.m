%% TODO:
%% replace for-loop by vectorizing the calculation
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
%% Theta1: hidden_layer_size x (input_layer_size + 1)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

%% Theta2: num_labels x (hidden_layer_size + 1)
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
% # of training examples
m = size(X, 1);

         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%% Note:
%% just as what you see from the NN architecture, the calculation here
%% makes sure every column in Ai(A1, A2, A3) is an input training example
%% for the next layer

%% X: m x input_layer_size
%% A1: (input_layer_size + 1) x m
%% add row x0=1
A1 = [ones(1, m); X'];

%% Z2: hidden_layer_size x m
Z2 = Theta1 * A1;

%% A2: hidden_layer_size x m
A2 = sigmoid(Z2);

%% add row a0=1
%% A2: (hidden_layer_size + 1) x m
A2 = [ones(1, m); A2];

%% Z3: num_labels x m
Z3 = Theta2 * A2;

%% A3: num_labels * m
A3 = sigmoid(Z3);


I = eye(num_labels);
for t = 1:m
    a3 = A3(:, t);
    y3 = I(:, y(t));
    J += -y3' * log(a3) - (1 - y3') * log(1 - a3);
    d3 = a3 - y3;

    z2 = Z2(:, t);
    a2 = A2(:, t);
    dgz = a2 .* (1 - a2);
    d2 = Theta2' * d3 .* dgz;

    a1 = A1(:, t);

    Theta2_grad += d3 * a2';
    Theta1_grad += d2(2:end) * a1';
end

J /= m;
Theta1_grad /= m;
Theta2_grad /= m;

%% do not penalize theta zero                                           
theta1_reg = Theta1;
theta1_reg(:, 1) = 0
theta2_reg = Theta2;
theta2_reg(:, 1) = 0;

%% do regularization                   
J += (sum((theta1_reg .^ 2)(:)) + sum((theta2_reg .^ 2)(:))) * lambda / (2 * m)
Theta1_grad += (lambda / m) * theta1_reg;
Theta2_grad += (lambda / m) * theta2_reg;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
