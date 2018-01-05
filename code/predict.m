function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

x1 = [ones(size(X, 1), 1) X];
pred1 = Theta1 * x1';
pred1 = pred1';

x2 = [ones(size(pred1, 1), 1) sigmoid(pred1)];
pred2 = x2 * Theta2';
pred3 = sigmoid(pred2);
[max_no, idx_no] = max(pred3, [], 2);
p = idx_no;

end
