"""% This function trains a neural network language model."""
from numpy import ones  # ones(shape) - shape of array e.g. 2 or (2,3) 
from numpy import zeros, eye
from numpy import exp, log, multiply
from numpy import dot, tile, ravel, asarray
from numpy.matlib import randn  # randn(shape)
from load_data import load_data
from fprop import fprop


#function [model] = train(epochs)
def train(epochs):
	"""
% Inputs:
%   epochs: Number of epochs to run.
% Output:
%   model: A struct containing the learned weights and biases and vocabulary.
	"""
	"""
if size(ver('Octave'),1)
  OctaveMode = 1;
  warning('error', 'Octave:broadcast');
  start_time = time;
else
  OctaveMode = 0;
  start_time = clock;
end
	"""

#% SET HYPERPARAMETERS HERE.
	batchsize = 100;  #% Mini-batch size.
	learning_rate = 0.1;  #% Learning rate; default = 0.1.
	momentum = 0.9;  #% Momentum; default = 0.9.
	numhid1 = 50;  #% Dimensionality of embedding space; default = 50.
	numhid2 = 200;  #% Number of units in hidden layer; default = 200.
	init_wt = 0.01;  #% Standard deviation of the normal distribution
                 #% which is sampled to get the initial weights; default = 0.01

#% VARIABLES FOR TRACKING TRAINING PROGRESS.
	show_training_CE_after = 100;
	show_validation_CE_after = 1000;

#% LOAD DATA.
#[train_input, train_target, valid_input, valid_target, ...
#  test_input, test_target, vocab] = load_data(batchsize);
	train_input, train_target, valid_input, valid_target, test_input, test_target, vocab = load_data(batchsize)

#[numwords, batchsize, numbatches] = size(train_input); 
	numwords, batchsize, numbatches = train_input.shape  # 3, 100, 3725
#% Size(vector, [dimension requried]) - get size of first dimension (rows)
#vocab_size = size(vocab, 2);
	vocab_size = vocab.shape[1]  # 250
	#print(numwords, batchsize, numbatches, vocab_size)
	
#% INITIALIZE WEIGHTS AND BIASES.
#% randn(rows, cols) - random matrix with zero mean and variance one
	# randn seems to produce a matrix instead of an array --> convert!!
	word_embedding_weights = init_wt * asarray(randn(vocab_size, numhid1));
	embed_to_hid_weights = init_wt * asarray(randn(numwords * numhid1, numhid2));
	hid_to_output_weights = init_wt * asarray(randn(numhid2, vocab_size));
	hid_bias = zeros((numhid2, 1));
	output_bias = zeros((vocab_size, 1));

	word_embedding_weights_delta = zeros((vocab_size, numhid1));
	word_embedding_weights_gradient = zeros((vocab_size, numhid1));
	embed_to_hid_weights_delta = zeros((numwords * numhid1, numhid2));
	hid_to_output_weights_delta = zeros((numhid2, vocab_size));
	hid_bias_delta = zeros((numhid2, 1));
	output_bias_delta = zeros((vocab_size, 1));
	expansion_matrix = eye((vocab_size));
	count = 0;
	tiny = exp(-30);
	
#% TRAIN.
#for epoch = 1:epochs
	for epoch in range(epochs):
#  fprintf(1, 'Epoch %d\n', epoch);
		print('Epoch %d\n' % (epoch+1))  # don't forget offset later on!
		this_chunk_CE = 0;
		trainset_CE = 0;
  #% LOOP OVER MINI-BATCHES.
  #for m = 1:numbatches
		for m in range(numbatches):
    #input_batch = train_input(:, :, m);
			input_batch = train_input[:, :, m]
    #target_batch = train_target(:, :, m);
			target_batch = train_target[:, :, m]

    #% FORWARD PROPAGATE.
    #% Compute the state of each layer in the network given the input batch
    #% and all weights and biases
    #[embedding_layer_state, hidden_layer_state, output_layer_state] = ...
    #  fprop(input_batch, ...
    #        word_embedding_weights, embed_to_hid_weights, ...
    #        hid_to_output_weights, hid_bias, output_bias);
			embedding_layer_state, hidden_layer_state, output_layer_state = fprop(input_batch,
				word_embedding_weights,
				embed_to_hid_weights,
				hid_to_output_weights,
				hid_bias,
				output_bias)
			
			# test for batch 5 word 0
			if m==5:
				test_words = input_batch[:, 0]
				w1 = word_embedding_weights[test_words[0]-1, 0:5]
				w2 = word_embedding_weights[test_words[1]-1, 0:5]
				w3 = word_embedding_weights[test_words[2]-1, 0:5]
				s1 = embedding_layer_state[0:5, 0]
				s2 = embedding_layer_state[50:55, 0]
				s3 = embedding_layer_state[100:105, 0]
				
				#print(test_words, '\n', w1, w2, w3, '\n', s1, s2, s3)

    #% COMPUTE DERIVATIVE.
    #%% Expand the target to a sparse 1-of-K vector.
    #expanded_target_batch = expansion_matrix(:, target_batch);
			expanded_target_batch = expansion_matrix[:, ravel(target_batch)-1]
			#if m==5: print(expansion_matrix.shape, target_batch.shape, expanded_target_batch.shape)
    #%% Compute derivative of cross-entropy loss function.
			# dE/dZout
			error_deriv = output_layer_state - expanded_target_batch;

    #% MEASURE LOSS FUNCTION.
    #CE = -sum(sum(...
    #  expanded_target_batch .* log(output_layer_state + tiny))) / batchsize;
			CE = -(expanded_target_batch
				* log(output_layer_state + tiny)).sum() / batchsize
			count =  count + 1
			this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count
			trainset_CE = trainset_CE + (CE - trainset_CE) / (m + 1)
    #fprintf(1, '\rBatch %d Train CE %.3f', m, this_chunk_CE);
			print('\rBatch %d Train CE %.3f' % (m, this_chunk_CE))
			
    #if mod(m, show_training_CE_after) == 0
			if (m + 1) % show_training_CE_after == 0:
      #fprintf(1, '\n');
				print('\n')
				count = 0;
				this_chunk_CE = 0;
    #end
    #if OctaveMode
    #  fflush(1);
    #end

    #% BACK PROPAGATE.
    #%% OUTPUT LAYER.
    #hid_to_output_weights_gradient =  hidden_layer_state * error_deriv';
			# dE/dWho
			hid_to_output_weights_gradient =  dot(hidden_layer_state, error_deriv.T)
    #output_bias_gradient = sum(error_deriv, 2);
			# use reshape to force 2D array from sum
			output_bias_gradient = error_deriv.sum(axis=1).reshape(-1, 1)
    #back_propagated_deriv_1 = (hid_to_output_weights * error_deriv) ...
			# dE/dYh
			back_propagated_deriv_1 = dot(hid_to_output_weights,
				error_deriv) * hidden_layer_state * (1 - hidden_layer_state)
    #%% HIDDEN LAYER.
    #% FILL IN CODE. Replace the line below by one of the options.
    #embed_to_hid_weights_gradient = zeros(numhid1 * numwords, numhid2);
    #% Options:
    #% (a) embed_to_hid_weights_gradient = back_propagated_deriv_1' * embedding_layer_state;
    #% (b) embed_to_hid_weights_gradient = embedding_layer_state * back_propagated_deriv_1';
			embed_to_hid_weights_gradient = dot(embedding_layer_state,
				back_propagated_deriv_1.T)
    #% (c) embed_to_hid_weights_gradient = back_propagated_deriv_1;
    #% (d) embed_to_hid_weights_gradient = embedding_layer_state;

    #% FILL IN CODE. Replace the line below by one of the options.
    #hid_bias_gradient = zeros(numhid2, 1);
    #% Options
    #% (a) hid_bias_gradient = sum(back_propagated_deriv_1, 2);
			# use reshape to force 2D array from sum
			hid_bias_gradient = back_propagated_deriv_1.sum(axis=1).reshape(-1, 1)
    #% (b) hid_bias_gradient = sum(back_propagated_deriv_1, 1);
    #% (c) hid_bias_gradient = back_propagated_deriv_1;
    #% (d) hid_bias_gradient = back_propagated_deriv_1';

    #% FILL IN CODE. Replace the line below by one of the options.
    #back_propagated_deriv_2 = zeros(numhid2, batchsize);
    #% Options
    #% (a) back_propagated_deriv_2 = embed_to_hid_weights * back_propagated_deriv_1;
			# dE/dZe
			back_propagated_deriv_2 = dot(embed_to_hid_weights,
				back_propagated_deriv_1)
    #% (b) back_propagated_deriv_2 = back_propagated_deriv_1 * embed_to_hid_weights;
    #% (c) back_propagated_deriv_2 = back_propagated_deriv_1' * embed_to_hid_weights;
    #% (d) back_propagated_deriv_2 = back_propagated_deriv_1 * embed_to_hid_weights';

    #word_embedding_weights_gradient(:) = 0;
			word_embedding_weights_gradient.fill(0)
    #%% EMBEDDING LAYER.
    #for w = 1:numwords
			for w in range(numwords):
       #word_embedding_weights_gradient = word_embedding_weights_gradient + ...
       #  expansion_matrix(:, input_batch(w, :)) * ...
       #  (back_propagated_deriv_2(1 + (w - 1) * numhid1 : w * numhid1, :)');
				word_embedding_weights_gradient = (
					word_embedding_weights_gradient +
					dot(expansion_matrix[:, ravel(input_batch[w, :])-1],
					(back_propagated_deriv_2[
						w * numhid1 : (w + 1) * numhid1, :].T)))
    #end
    
    #% UPDATE WEIGHTS AND BIASES.
    #word_embedding_weights_delta = ...
    #  momentum .* word_embedding_weights_delta + ...
    #  word_embedding_weights_gradient ./ batchsize;
			word_embedding_weights_delta = (momentum *
				word_embedding_weights_delta +
				word_embedding_weights_gradient /
				batchsize)
    #word_embedding_weights = word_embedding_weights...
    #  - learning_rate * word_embedding_weights_delta;
			word_embedding_weights = (word_embedding_weights - 
				learning_rate * word_embedding_weights_delta)

    #embed_to_hid_weights_delta = ...
    #  momentum .* embed_to_hid_weights_delta + ...
    #  embed_to_hid_weights_gradient ./ batchsize;
			embed_to_hid_weights_delta = (momentum *
				embed_to_hid_weights_delta +
				embed_to_hid_weights_gradient /
				batchsize)
    #embed_to_hid_weights = embed_to_hid_weights...
    #  - learning_rate * embed_to_hid_weights_delta;
			embed_to_hid_weights = (embed_to_hid_weights -
				learning_rate * embed_to_hid_weights_delta)

    #hid_to_output_weights_delta = ...
    #  momentum .* hid_to_output_weights_delta + ...
    #  hid_to_output_weights_gradient ./ batchsize;
			hid_to_output_weights_delta = (momentum *
				hid_to_output_weights_delta +
				hid_to_output_weights_gradient /
				batchsize)
    #hid_to_output_weights = hid_to_output_weights...
    #  - learning_rate * hid_to_output_weights_delta;
			hid_to_output_weights = (hid_to_output_weights -
				learning_rate * hid_to_output_weights_delta)

    #hid_bias_delta = momentum .* hid_bias_delta + ...
    #  hid_bias_gradient ./ batchsize;
			#print(hid_bias_delta.shape, hid_bias_gradient.shape)
			hid_bias_delta = (momentum *
				hid_bias_delta +
				hid_bias_gradient / batchsize)
    #hid_bias = hid_bias - learning_rate * hid_bias_delta;
			hid_bias = hid_bias - learning_rate * hid_bias_delta

    #output_bias_delta = momentum .* output_bias_delta + ...
    #  output_bias_gradient ./ batchsize;
			output_bias_delta = (momentum * output_bias_delta +
				output_bias_gradient / batchsize)
    #output_bias = output_bias - learning_rate * output_bias_delta;
			output_bias = output_bias - learning_rate * output_bias_delta

    #% VALIDATE.
    #if mod(m, show_validation_CE_after) == 0
			if (m +1) % show_validation_CE_after == 0:
      #fprintf(1, '\rRunning validation ...');
				print('\rRunning validation ...')
      #if OctaveMode
        #fflush(1);
      #end
      #[embedding_layer_state, hidden_layer_state, output_layer_state] = ...
      #  fprop(valid_input, word_embedding_weights, embed_to_hid_weights,...
      #        hid_to_output_weights, hid_bias, output_bias);
				embedding_layer_state, hidden_layer_state, output_layer_state = fprop(
					valid_input,
					word_embedding_weights,
					embed_to_hid_weights,
					hid_to_output_weights,
					hid_bias,
					output_bias)
      #datasetsize = size(valid_input, 2);
				datasetsize = valid_input.shape[1]
      #expanded_valid_target = expansion_matrix(:, valid_target);
				expanded_valid_target = expansion_matrix[:,
					ravel(valid_target)-1]
      #CE = -sum(sum(...
      #  expanded_valid_target .* log(output_layer_state + tiny))) /datasetsize;
				CE = -(expanded_valid_target
					* log(output_layer_state + tiny)).sum() / datasetsize
      #fprintf(1, ' Validation CE %.3f\n', CE);
				print(' Validation CE %.3f\n' % (CE))
      #if OctaveMode
      #  fflush(1);
      #end
    #end
  #end
  #fprintf(1, '\rAverage Training CE %.3f\n', trainset_CE);
		print('\rAverage Training CE %.3f\n' % (trainset_CE))
#end

#fprintf(1, 'Finished Training.\n');
	print('Finished Training.\n')
#if OctaveMode
#  fflush(1);
#end
#fprintf(1, 'Final Training CE %.3f\n', trainset_CE);
	print('Final Training CE %.3f\n' % (trainset_CE))

#% EVALUATE ON VALIDATION SET.
#fprintf(1, '\rRunning validation ...');
	print('\rRunning validation ...')
#if OctaveMode
#  fflush(1);
#end
#[embedding_layer_state, hidden_layer_state, output_layer_state] = ...
#  fprop(valid_input, word_embedding_weights, embed_to_hid_weights,...
#        hid_to_output_weights, hid_bias, output_bias);
	print('Validation input shape: {}'.format(valid_input.shape))
	embedding_layer_state, hidden_layer_state, output_layer_state = fprop(
		valid_input,
		word_embedding_weights,
		embed_to_hid_weights,
		hid_to_output_weights,
		hid_bias,
		output_bias)
#datasetsize = size(valid_input, 2);
	datasetsize = valid_input.shape[1]
#expanded_valid_target = expansion_matrix(:, valid_target);
	expanded_valid_target = expansion_matrix[:, ravel(valid_target)-1]
#CE = -sum(sum(...
#  expanded_valid_target .* log(output_layer_state + tiny))) / datasetsize;
	CE = -(expanded_valid_target
		* log(output_layer_state + tiny)).sum() / datasetsize
#fprintf(1, '\rFinal Validation CE %.3f\n', CE);
	print('\rFinal Validation CE %.3f\n' % (CE))
#if OctaveMode
#  fflush(1);
#end
	
	# reset states to avoid running out of memory on raspberry pi!
	embedding_layer_state, hidden_layer_state, output_layer_state = 0, 0, 0
#% EVALUATE ON TEST SET.
#fprintf(1, '\rRunning test ...');
	print('\rRunning test ...')
#if OctaveMode
#  fflush(1);
#end
#[embedding_layer_state, hidden_layer_state, output_layer_state] = ...
#  fprop(test_input, word_embedding_weights, embed_to_hid_weights,...
#        hid_to_output_weights, hid_bias, output_bias);
	print('Test input shape: {}'.format(test_input.shape))
	embedding_layer_state, hidden_layer_state, output_layer_state = fprop(
		test_input,
		word_embedding_weights,
		embed_to_hid_weights,
		hid_to_output_weights,
		hid_bias,
		output_bias)
#datasetsize = size(test_input, 2);
	datasetsize = test_input.shape[1]
#expanded_test_target = expansion_matrix(:, test_target);
	expanded_test_target = expansion_matrix[:, ravel(test_target)-1]
#CE = -sum(sum(...
#  expanded_test_target .* log(output_layer_state + tiny))) / datasetsize;
	CE = -(expanded_test_target
		* log(output_layer_state + tiny)).sum() / datasetsize
#fprintf(1, '\rFinal Test CE %.3f\n', CE);
	print('\rFinal Test CE %.3f\n' % (CE))
#if OctaveMode
#  fflush(1);
#end

	"""
model.word_embedding_weights = word_embedding_weights;
model.embed_to_hid_weights = embed_to_hid_weights;
model.hid_to_output_weights = hid_to_output_weights;
model.hid_bias = hid_bias;
model.output_bias = output_bias;
model.vocab = vocab;

% In MATLAB replace line below with 'end_time = clock;'
if OctaveMode
  end_time = time;
  diff = end_time - start_time;
else
  end_time = clock;
  diff = etime(end_time, start_time);
end
fprintf(1, 'Training took %.2f seconds\n', diff);
end
	"""
if __name__ == '__main__':
	train(10)
