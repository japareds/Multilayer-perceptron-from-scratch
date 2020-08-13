function p = prediction(Theta_matrix,X,L)
% Predict label of input examples
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
  [max_h,p] = max(hyp); 
  
end
