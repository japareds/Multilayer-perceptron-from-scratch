function [J,grad_J] = FF_costFunction(Theta_matrix_vector,input_layer_length,hidden_layer_length,output_layer_length,L,X,y,lambda,m)
  % cost function and gradient definitions
  J = 0;
  %J_grad = {};% partial derivative dJ / dTheta_i
  %for i = 1:size(Theta_matrix,3)
  %  J_grad(:,:,i) = zeros(size(Theta_matrix{:,:,i}));
  %end
  
  % ========== reshape Theta matrix vector to n matrices ==========
  Theta_matrix = {};
  numel_hidden_matrices = hidden_layer_length*(hidden_layer_length+1);% matrices i = 2:L-2 have the same size
  Theta_matrix(:,:,1) = reshape( Theta_matrix_vector(1:(hidden_layer_length*(input_layer_length+1))) , hidden_layer_length,input_layer_length+1);
  if L != 3 
    Theta_matrix(:,:,2) = reshape( Theta_matrix_vector( numel(Theta_matrix{:,:,1}) + 1  : numel(Theta_matrix{:,:,1}) + numel_hidden_matrices) , hidden_layer_length,hidden_layer_length+1);    
    for i = 3:L-2
      Theta_matrix(:,:,i) = reshape( Theta_matrix_vector((numel(Theta_matrix{:,:,1}) + (i-2)*numel_hidden_matrices + 1) : ( numel(Theta_matrix{:,:,1})+ (i-1)*numel_hidden_matrices )), hidden_layer_length,hidden_layer_length+1 );
    end
  end  
  Theta_matrix(:,:,L-1) = reshape( flip(Theta_matrix_vector(end:-1:(end-(output_layer_length*(hidden_layer_length+1))+1) )), output_layer_length,hidden_layer_length+1);
  % ========== activating nodes ==========
  %  - remember to add bias node (+1 in first entry)
  a = {};
  % initializing all layers for all examples iterations
  a(:,:,1) = [ones(1,size(X,2));X];
  for i = 2:L-1
    a(:,:,i) = [ones(1,size(X,2));sigmoid(Theta_matrix{:,:,i-1}(:,:) * a{:,:,i-1}(:,:))];
  end
  a(:,:,L) = sigmoid(Theta_matrix{:,:,L-1}(:,:) * a{:,:,L-1}(:,:)); %output layer activation: classification hypothesis for all examples
  hyp = a{:,:,L};
  % true label vector for every example
  t = zeros(output_layer_length,size(X,2));%true label in last layer
  for i = 1:size(t,2)
    t(y(i),i) = 1;
  end
  %========== compute cost ==========
  cost = -t.*log(hyp) - (1-t).*log(1-hyp);
  %Theta matrices without bias
  Theta_NoBias = {};
  TNB = zeros(size(Theta_matrix,3),1);%vector containing the sum of all items in each Theta Matrix (no bias)
  for i = 1:size(Theta_matrix,3)
    Theta_NoBias(:,:,i) = Theta_matrix{:,:,i}(:,2:end);
    TNB(i) = sum(sum(Theta_NoBias{:,:,i}(:,:).^2));
  end
  J = (1.0/size(X,2)) * sum(sum(cost)) + (lambda/(2*size(X,2))) * sum(TNB);
  % ========== gradient ==========
  % backpropagation algorithm
  deltas = cell(1,1,L); %error term of every node in every layer for all examples
  deltas(:,:,L) = a{:,:,L} - t;%error term output layer for all examples
  for i = L:-1:2
    %fprintf('\n computing delta=%d \n',i-1)
    if i == L
      deltas(:,:,i-1) = ( Theta_matrix{:,:,i-1}(:,:)' * deltas{:,:,i} ).*( a{:,:,i-1} ).*( 1.0 - a{:,:,i-1} );
    else
      deltas(:,:,i-1) = ( Theta_matrix{:,:,i-1}(:,:)' * deltas{:,:,i}(2:end,:) ).*( a{:,:,i-1} ).*( 1.0 - a{:,:,i-1} );
    end
    %fprintf('\n done \n')
  end
  %fprintf('\n Gradient computing takes time... \n Press enter to continue \n')
  %pause;
  dJ_dTheta = cell(1,1,size(Theta_matrix,3));  
  for i = 1:size(dJ_dTheta,3)
    %fprintf('\n Empezando con gradiente entrada %d / %d \n ',i,size(dJ_dTheta,3))
    dJ_dTheta{:,:,i} = zeros(size(Theta_matrix{:,:,i}));  
    for j = 1:m %iterate for every example
      if mod(j,1000) == 0
     %   fprintf('\n van 1000 mas \n')
      end
      % consider case if Theta L-1 that uses delta L
      % delta for every other layer has +1 bias
      if i != L-1
        dJ_dTheta(:,:,i) = dJ_dTheta{:,:,i} + deltas{:,:,i+1}(2:end,j)*a{:,:,i}(:,j)';
      end
      if i == L-1
        dJ_dTheta(:,:,i) = dJ_dTheta{:,:,i} + deltas{:,:,L}(:,j)*a{:,:,i}(:,j)';
      end
    end
    % regularized dJ/dTheta
    % note to not regularize 1st column: bias term  
    dJ_dTheta(:,:,i) = (1/m) * dJ_dTheta{:,:,i} + (lambda/m) * [zeros(size(Theta_NoBias{:,:,i},1),1) Theta_NoBias{:,:,i}];
  end
  %return gradient vector for optimization
  grad_J = [];
  for i = 1:size(dJ_dTheta,3)
    grad_J = [grad_J(:) ; dJ_dTheta{:,:,i}(:)];
  end
end
