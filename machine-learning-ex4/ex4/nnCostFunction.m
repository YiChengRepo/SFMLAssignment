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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
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

% ==================================================================================
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%Theta1 25 401
%Theta2 10 26
%hidden_layer_size 25
%X 5000 400
%forward propagation to calcualte h(X) which is also a(3), the output layer 

%1.0 cost

%layer 1
%a1 5000 401
a1 = [ones(m,1) X];
%z2 5000 25
z2 = a1*Theta1';

%z2 5000 25
a2 = sigmoid(z2);
%z2 5000 26
a2 = [ones(m,1) a2];

%5000*10
z3 = a2*Theta2';
a3 = sigmoid(z3);

%convert the 5000*1's y vector into a 10*5000 matrix
%where each column is a vector represent the original value
yk = zeros(num_labels, m);
for i=1:m
    %for each column, set the orignal value to be 1
    %for example, 1,2,5
    % then yk(1,1) = 1, yk(2,2) = 1, yk(5,3) = 1
    yk(y(i),i) = 1;
end
% yk

%(10*5000)' dot product  (5000*10)
%and result is till a 5000*10
%dot product does element wise multiplication 
% j_matrix = (-yk)' .* log(a3) - (1-yk)' .* log(1-a3)
% j_vector = sum(j_matrix)
% j_value = sum(j_vector)
% J = 1/m * j_value

J = 1/m * sum(sum( (-yk)' .* log(a3) - (1-yk)' .* log(1-a3) ));

%1.1 regularization
regulator = (lambda /(2*m)) * ( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );

J = J + regulator;

%Part 2 ==================================================================================
%back propagation

%Theta1 25 401
%Theta2 10 26
%hidden_layer_size 25
%X 5000 400

%compute for each training example 
for t = 1:m
    %401 * 1
    a1 = [1; X(t,:)'];
    %z2 25 * 1 
    z2 = Theta1*a1;

    %z2 25 1
    a2 = sigmoid(z2);
    %26 1
    a2 = [1; a2];

    %10*1
    z3 = Theta2*a2;
    a3 = sigmoid(z3);
    
   
    %yk(:,t) is a 10 * 1 vector 
    %so a3 is 10*1
    d3 = a3 - yk(:,t);
    
    %26*10 * 10*1 = 26*1
    d2 = (Theta2' * d3) .* [1; sigmoidGradient(z2)];
    %25*1
    d2 = d2(2:end); % no need for bias row for delta
    
    % Big delta accumulate 
    %d2 * a1' = 25*1 by 1*401 = 25*401 which is the same size as Theta1, so it can do +
    %d3 * a2' = 10*1 by 1*26 = 10*26 which is the same size as Theta2, so it can do +
    Theta1_grad = Theta1_grad + d2 * a1';
    Theta2_grad = Theta2_grad + d3 * a2';

    
end    

%not regularized
% Theta1_grad = (1/m) * Theta1_grad;
% Theta2_grad = (1/m) * Theta2_grad;


%regularized
Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) ./ m + ((lambda/m) * Theta1(:, 2:end));


Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) ./ m + ((lambda/m) * Theta2(:, 2:end));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
