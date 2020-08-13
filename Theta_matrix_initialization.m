function [Theta] = Theta_matrix_initialization(L,input_layer_length,hidden_layer_length,output_layer_length)
  Theta = {};
  epsilon = 0.12;
  fprintf('\n Initializing network weights \n Intial weights between [-%d,%d] \n',epsilon,epsilon);
  Theta(:,:,1) = rand(hidden_layer_length,input_layer_length+1)*(2*epsilon - epsilon);%Input layer theta matrix. Each row corresponds to Theta' to activate node i
  for i = 2:L-2
    Theta(:,:,i) = rand(hidden_layer_length,hidden_layer_length+1)*(2*epsilon - epsilon);%hidden layers theta matrices
  end
  Theta(:,:,L-1) = rand(output_layer_length,hidden_layer_length+1)*(2*epsilon - epsilon);%last hidden layer theta matrix
end