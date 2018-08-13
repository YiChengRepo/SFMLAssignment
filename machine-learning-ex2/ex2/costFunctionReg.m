function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
num_f = size(X, 2);
grad = zeros(num_f, 1);

z = X*theta;
h = zeros(size(z, 1),1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(z);
%adding theta 2 to end excluding theta 0
J = ((1/m) * sum(-y'*log(h) - (1-y)'*log(1-h))) + (lambda/(2*m))*sum(theta(2:length(theta)).*theta(2:length(theta)));

for i=1:num_f
    %no regularization for theta 0
    if i == 1
        grad(i) = (1/m) * sum((h - y).*X(:,i));
    else
        grad(i) = (1/m) * sum((h - y).*X(:,i)) + lambda/m * theta(i);
end;

% =============================================================

end
