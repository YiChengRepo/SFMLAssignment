function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);



% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

%adding bias to column 1 to input layer matrix X, 500 by 401 matrix
RX = [ones(m ,1), X];
%output 5000 by 25 matrix
layer2 = sigmoid(RX * Theta1');
%adding bias to column 1 to layer 2 matrix, preparation input for layer 3
%output 5000 by 26 matrix
layer2 = [ones(m, 1), layer2];
%output 5000 by 10 matrix
layer3 = sigmoid(layer2 * Theta2');
[val,p] = max(layer3, [], 2);
% =========================================================================

end
