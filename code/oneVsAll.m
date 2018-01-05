function [all_theta] = oneVsAll(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

X = [ones(m, 1) X];

options = optimset('GradObj', 'on', 'MaxIter', 500);
initial_theta = zeros(n+1, 1);
for c = 1:num_labels,
    [theta] = fmincg(@(t) (lrCostFunction(t, X, (y==c), lambda)), initial_theta, options);
    all_theta(c, :) = theta';
end

end
