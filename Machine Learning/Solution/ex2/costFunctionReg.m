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

theta_mod = theta;
theta_mod(1) = 0;

regular_grad = ((lambda/m) * theta_mod);
regular_J = sum((lambda/(2*m))*(theta_mod).^2);

grad = ((1/m)*( X' * (sigmoid(X * theta) - y))) + regular_grad;

cost = (-y' * log(sigmoid(X * theta))- ((1-y)' * log(1-(sigmoid(X * theta)))));
J =  sum((1/m) * cost) + regular_J;

% =============================================================

end
