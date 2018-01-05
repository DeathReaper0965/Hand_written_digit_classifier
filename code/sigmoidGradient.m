function g = sigmoidGradient(z)

g = zeros(size(z));
g = (1 ./ (1 + exp(-z))) .* (1 - (1 ./ (1 + exp(-z))));

end
