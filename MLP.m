%% ========== Neural Network from scratch ========== 
% The idea is to programm a NN that classifies m examples with n features each 
% in order to understand how does it work.
% The hidden layers are represented by a [r x c] matrix depending in the number 
% of nodes in every layer.
% The number of hidden layers can be introduced by user, so the matrices used
% will be stored in a cell structure {:,:,i} and don't have to add a new variable
% per layer. I could change that later to make the code computationally efficient.
%

clear all; close all; clc

%% ========== Load data: handwritten digits ==========
% data matrix consists has (n,m) dimensions
% n: num of features
% m: num of examples
load('ex4data1.mat');
n_imgs = 2; %num imgs to show = n_imgs^2
fprintf('\n %d images loaded \n Each example is a %d x %d pix grayscale image',size(X,1),sqrt(size(X,2)),sqrt(size(X,2)));
fprintf('\n Randomly displaying %d images\n',n_imgs^2)
figure(1)
colormap(gray)
for i = 1:n_imgs^2
  subplot(n_imgs,n_imgs,i)  
  imagesc(reshape(X(round(rand()*size(X,1)),:),sqrt(size(X,2)),sqrt(size(X,2))))
end

fprintf('\nPress enter to continue\n');
pause;

%% ========== Network parameters ==========
% The network consists of N layers: 
%  -> 1 input layer (containing features)
%  -> 1 output layer (class hypothesis)
%  -> N-2 hidden layers
% The input received by the network is a [n x m] matrix:
%  -> n: number of features
%  -> m: number of examples
n = 400; %num of features
if size(X,1) != n
  X = X';
end
m = size(X,2); %num of examples

fprintf('\n Neural Netowrk will be used to classify handwritten digits 0-9 \n');
L = 3;%num of layers
input_layer_length = size(X,1);%length of feature vector 
hidden_layer_length = input('Insert number of nodes in hidden layers: ');%num of nodes in hidden layers
output_layer_length = numel(unique(y));%num of labels
fprintf('\n Neural Network info:\n %d layers \n %d features \n Every hidden layer has the same amount of neurons: %d neurons in hidden layers \n %d labels\n',L,input_layer_length,hidden_layer_length,output_layer_length);

%Initialize network weights
load_matrices = 0; % 0 or 1. 0 = don't load matrices, random initialize them. 1 = load matrices from files
if load_matrices == 0 
  initial_Theta_matrix = Theta_matrix_initialization(L,input_layer_length,hidden_layer_length,output_layer_length);
  fprintf('\n Weights matrices were initialized \n Press enter to continue \n')
  pause;
else
  fprintf('\n Loading weights matrices from files \n Number of nodes in hidden layers MUST be = 25 ')
  hidden_layer_length = 25;
  L = 3;%num of layers
  load('ex4weights.mat');
  initial_Theta_matrix = {};
  initial_Theta_matrix(:,:,1) = Theta1;
  initial_Theta_matrix(:,:,2) = Theta2;
end  

%% ========== Training NN ==========
% NN feedforward propagation and cost computing
lambda = 1;
initial_Theta_matrix_vector = [];
for i = 1:size(initial_Theta_matrix,3)
initial_Theta_matrix_vector = [initial_Theta_matrix_vector(:);initial_Theta_matrix{:,:,i}(:)];
end

[J,grad_J] = FF_costFunction(initial_Theta_matrix_vector,input_layer_length,hidden_layer_length,output_layer_length,L,X,y,lambda,m);
fprintf('\n Cost at initial parameters: %f \n Training Neural Network \n.',J)
fprintf('\n Optimizer used to efficiently train cost function: fminunc \n Press Enter to continue \n')
pause;
% Note that with the right learning settings the NN could perfectly fit the training set
% optimizer needs a parameters vector: append all matrices in one single vector (:,1)
options = optimset('MaxIter', 50);
costFunction = @(p) FF_costFunction(p,input_layer_length,hidden_layer_length,output_layer_length,L,X,y,lambda,m);
[weights, costs] = fmincg(costFunction,initial_Theta_matrix_vector, options);
% Transform weights to matrices
Theta_matrix = {};
numel_hidden_matrices = hidden_layer_length*(hidden_layer_length+1);% matrices i = 2:L-2 have the same size
Theta_matrix(:,:,1) = reshape( weights(1:(hidden_layer_length*(input_layer_length+1))) , hidden_layer_length,input_layer_length+1);
if L!= 3
  Theta_matrix(:,:,2) = reshape( weights( numel(Theta_matrix{:,:,1}) + 1  : numel(Theta_matrix{:,:,1}) + numel_hidden_matrices) , hidden_layer_length,hidden_layer_length+1);
  for i = 3:L-2
    Theta_matrix(:,:,i) = reshape( weights((numel(Theta_matrix{:,:,1}) + (i-2)*numel_hidden_matrices + 1) : ( numel(Theta_matrix{:,:,1})+ (i-1)*numel_hidden_matrices )), hidden_layer_length,hidden_layer_length+1 );
  end
end

Theta_matrix(:,:,L-1) = reshape( flip(weights(end:-1:(end-(output_layer_length*(hidden_layer_length+1))+1) )), output_layer_length,hidden_layer_length+1);

fprintf('\n minimum cost function found: %f \n Press enter to continue. \n',costs(end))
pause;
% ========== Predictions ==========
predicted_label = prediction(Theta_matrix, X, L)';
TP = (numel(find(predicted_label == y)))/(m);
fprintf('\nTraining Set Accuracy: %d \n', TP*100);


