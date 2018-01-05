function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
theta_shift = [0; theta(2:end, :)];

J = ((1/m) * ((-y' * log(h)) - ((1-y)' * log(1-h)))) + (lambda/(2*m) * (theta_shift' * theta_shift));

grad = ((1/m) * X' * (h-y)) + ((lambda/m) * theta_shift);
grad = grad(:);

end
