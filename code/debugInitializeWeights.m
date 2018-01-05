function W = debugInitializeWeights(fan_out, fan_in)

W = zeros(fan_out, 1 + fan_in);

W = reshape(sin(1:numel(W)), size(W)) / 10;

end
