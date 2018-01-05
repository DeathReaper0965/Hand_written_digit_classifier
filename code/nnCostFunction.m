function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

I = eye(num_labels);
Y = zeros(m, num_labels);

for i = 1:m,
    Y(i, :) = I(y(i), :);
end

x1 = [ones(m, 1) X];
pred1 = Theta1 * x1';
pred1 = pred1';

x2 = [ones(size(pred1, 1), 1) sigmoid(pred1)];
pred2 = x2 * Theta2';
h = sigmoid(pred2);

shift1 = Theta1(:, 2:end);
Theta1_reg = [zeros(size(shift1, 1), 1) shift1];
shift2 = Theta2(:, 2:end);
Theta2_reg = [zeros(size(shift2, 1), 1) shift2];

%Compute cost...
J = (1/m) * sum(sum(-Y .* log(h) - (1-Y) .* log(1-h))) + (lambda/(2*m) * sum(sum(Theta1_reg .^ 2)));
J = J + (lambda/(2*m) * sum(sum(Theta2_reg .^ 2)));

%Backpropagation

delta3 = (h - Y)';
delta2 = (Theta2' * delta3) .* [ones(1, m); sigmoidGradient(pred1)'];


Theta2_grad = (1/m) * delta3 * x2 + (lambda/m) * [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)];
Theta1_grad = (1/m) * delta2(2:end, :) * x1 + (lambda/m) * [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)];

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
