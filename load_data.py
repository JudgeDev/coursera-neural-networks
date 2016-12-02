import math
import scipy.io  # for reading .mat files
#from numpy import 
#function [train_input, train_target, valid_input, valid_target, test_input, test_target, vocab] = load_data(N)
def load_data(N):
	"""
% This method loads the training, validation and test set.
% It also divides the training set into mini-batches.
% Inputs:
%   N: Mini-batch size.
% Outputs:
%   train_input: An array of size D X N X M, where
%                 D: number of input dimensions (in this case, 3).
%                 N: size of each mini-batch (in this case, 100).
%                 M: number of minibatches.
%   train_target: An array of size 1 X N X M.
%   valid_input: An array of size D X number of points in the validation set.
%   test: An array of size D X number of points in the test set.
%   vocab: Vocabulary containing index to word mapping.
	"""
#load data.mat;
	data = scipy.io.loadmat('data.mat')
#numdims = size(data.trainData, 1);
	numdims = data['data']['trainData'][0,0].shape[0]
	D = numdims - 1;
#M = floor(size(data.trainData, 2) / N);
	M = math.floor(data['data']['trainData'][0,0].shape[1] / N)
#train_input = reshape(data.trainData(1:D, 1:N * M), D, N, M);
	train_input = data['data']['trainData'][0,0][0:D, 0:N*M].reshape(D, N, M, order='F')
#train_target = reshape(data.trainData(D + 1, 1:N * M), 1, N, M);
	train_target = data['data']['trainData'][0,0][D, 0:N*M].reshape(1, N, M, order='F')
#valid_input = data.validData(1:D, :);
	valid_input = data['data']['validData'][0,0][0:D, :]
#valid_target = data.validData(D + 1, :);
	valid_target = data['data']['validData'][0,0][D, :]
#test_input = data.testData(1:D, :);
	test_input = data['data']['testData'][0,0][0:D, :]
#test_target = data.testData(D + 1, :);
	test_target = data['data']['testData'][0,0][D, :]
#vocab = data.vocab;
	vocab = data['data']['vocab'][0,0]
#end
	print('batch size: {}\ninput dimensions: {}\nnumber of batches: {}'.format(N, numdims, M)) 
	#print(data['data']['trainData'][0,0][0:4,0:100])
	#print(train_input[:,:,0])
	#print(train_target[:,:,0])
	#print(data['data']['vocab'][0,0][0,34])
	print('train_input: {}\ntrain_target: {}\nvalid_input: {}\nvalid_target: {}\n'.format(
		train_input.shape,
		train_target.shape,
		valid_input.shape,
		valid_target.shape))
	return train_input, train_target, valid_input, valid_target, test_input, test_target, vocab
