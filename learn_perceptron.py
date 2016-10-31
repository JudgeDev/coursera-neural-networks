"""%% Learns the weights of a perceptron and displays the results."""

#import numpy as np
# Numpy functions used:
from numpy import c_, r_  # add columns
from numpy import ones  # ones(shape) - shape of array e.g. 2 or (2,3) 
from numpy.matlib import randn  # randn(shape)
from numpy.matlib import array 
from numpy.linalg import norm  # norm(v) - L2 norm
from numpy import asscalar
from numpy import float32

import scipy.io  # for reading .mat files

from plot_perceptron import plot_perceptron

#function [w] = learn_perceptron(neg_examples_nobias,pos_examples_nobias,w_init,w_gen_feas)
def learn_perceptron(neg_examples_nobias,pos_examples_nobias,w_init,w_gen_feas):
	"""
%% 
% Learns the weights of a perceptron for a 2-dimensional dataset and plots
% the perceptron at each iteration where an iteration is defined as one
% full pass through the data. If a generously feasible weight vector
% is provided then the visualization will also show the distance
% of the learned weight vectors to the generously feasible weight vector.
% Required Inputs:
%   neg_examples_nobias - The num_neg_examples x 2 matrix for the examples with target 0.
%       num_neg_examples is the number of examples for the negative class.
%   pos_examples_nobias - The num_pos_examples x 2 matrix for the examples with target 1.
%       num_pos_examples is the number of examples for the positive class.
%   w_init - A 3-dimensional initial weight vector. The last element is the bias.
%   w_gen_feas - A generously feasible weight vector.
% Returns:
%   w - The learned weight vector.
%%
	"""
#%Bookkeeping
#% Size(vector, [dimension requried]) - get size of first dimension (rows)
#num_neg_examples = size(neg_examples_nobias,1);
	num_neg_examples = neg_examples_nobias.shape[0]
#num_pos_examples = size(pos_examples_nobias,1);
	num_pos_examples = pos_examples_nobias.shape[0]
	
	num_err_history = [];  # should be array?
	w_dist_history = [];  # should be array?

#%Here we add a column of ones to the examples in order to allow us to learn
#%bias parameters
#% Ones(rows, cols)
#neg_examples = [neg_examples_nobias,ones(num_neg_examples,1)];
	neg_examples = c_[neg_examples_nobias, ones(num_neg_examples)].astype(float32)
#pos_examples = [pos_examples_nobias,ones(num_pos_examples,1)];
	pos_examples = c_[pos_examples_nobias, ones(num_pos_examples)].astype(float32)

#%If weight vectors have not been provided, initialize them appropriately
#% exist(name, type)
#% || is short circuit boolean or (stops when overall value determined)
#% randn(rows, cols) - random matrix with zero mean and variance one
#if (~exist('w_init','var') || isempty(w_init))
	if 'w_init' not in locals() or w_init.size == 0:
#	w = randn(3,1);
		w = randn((3,1));
#else
	else:
		w = w_init.astype(float32)
#end
#if (~exist('w_gen_feas','var'))
	if 'w_gen_feas' not in locals():
		w_gen_feas = [];  # should be array?
#end

#%Find the data points that the perceptron has incorrectly classified
#%and record the number of errors it makes.
	iter_ = 0;
	[mistakes0, mistakes1] = eval_perceptron(neg_examples,pos_examples,w);
#num_errs = size(mistakes0,1) + size(mistakes1,1);
	num_errs = mistakes0.shape[0] + mistakes1.shape[0]

#% (..., end, ...) end index is last entry for a particular dimension
#num_err_history(end+1) = num_errs;
	num_err_history.append(num_errs)
#fprintf('Number of errors in iteration %d:\t%d\n',iter,num_errs);
	print('Number of errors in iteration %d:\t%d\n' % (iter_,num_errs));
#fprintf(['weights:\t', mat2str(w), '\n']);
	print('weights:\n', w, '\n');
	plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history);
#key = input('<Press enter to continue, q to quit.>', 's');
	key = input('<Press enter to continue, q to quit.>')
#if (key == 'q')
	if (key == 'q'):
		return;
#end

#%If a generously feasible weight vector exists, record the distance
#%to it from the initial weight vector.
#if (length(w_gen_feas) ~= 0)
	if w_gen_feas.size != 0:
    #% (..., end, ...) end index is last entry for a particular dimension
    #% norm(x1 - x2) is Euclidean distance between two vectors
    #w_dist_history(end+1) = norm(w - w_gen_feas);
		w_dist_history.append(norm(w - w_gen_feas))
#end

#%Iterate until the perceptron has correctly classified all points.
#while (num_errs > 0)
	while num_errs > 0:
		iter_ = iter_ + 1;

    #%Update the weights of the perceptron.
		w = update_weights(neg_examples,pos_examples,w);

    #%If a generously feasible weight vector exists, record the distance
    #%to it from the current weight vector.
    #if (length(w_gen_feas) ~= 0)
		if w_gen_feas.size != 0:
        #w_dist_history(end+1) = norm(w - w_gen_feas);
			w_dist_history.append(norm(w - w_gen_feas))
    #end
    #%Find the data points that the perceptron has incorrectly classified.
    #%and record the number of errors it makes.
		[mistakes0, mistakes1] = eval_perceptron(neg_examples,pos_examples,w);
	#num_errs = size(mistakes0,1) + size(mistakes1,1);
		num_errs = mistakes0.shape[0] + mistakes1.shape[0]
    #num_err_history(end+1) = num_errs;
		num_err_history.append(num_errs)
    #fprintf('Number of errors in iteration %d:\t%d\n',iter,num_errs);
		print('Number of errors in iteration %d:\t%d\n' % (iter_,num_errs));
    #fprintf(['weights:\t', mat2str(w), '\n']);
		print('weights:\n', w, '\n');
		plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history);
    #key = input('<Press enter to continue, q to quit.>', 's');
		key = input('<Press enter to continue, q to quit.>')
    #if (key == 'q')
		if (key == 'q'):
			break;
    #end
#end

#%WRITE THE CODE TO COMPLETE THIS FUNCTION
#function [w] = update_weights(neg_examples, pos_examples, w_current)
def update_weights(neg_examples, pos_examples, w_current):
	"""
%% 
% Updates the weights of the perceptron for incorrectly classified points
% using the perceptron update algorithm. This function makes one sweep
% over the dataset.
% Inputs:
%   neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
%       num_neg_examples is the number of examples for the negative class.
%   pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
%       num_pos_examples is the number of examples for the positive class.
%   w_current - A 3-dimensional weight vector, the last element is the bias.
% Returns:
%   w - The weight vector after one pass through the dataset using the perceptron
%       learning rule.
%%
	"""
#w = w_current;
	w = w_current
#num_neg_examples = size(neg_examples,1);
	num_neg_examples = neg_examples.shape[0]
#num_pos_examples = size(pos_examples,1);
	num_pos_examples = pos_examples.shape[0]
#for i=1:num_neg_examples
	for i in range(num_neg_examples):
    #this_case = neg_examples(i,:);
		this_case = neg_examples[i,:]
    #x = this_case'; %Hint
		x = this_case.T  #%Hint
    #activation = this_case*w;
		activation = asscalar(this_case.dot(w))
    #if (activation >= 0)  % incorrect  1 ouput
		if activation >= 0:  #% incorrect  1 ouput
        #%YOUR CODE HERE
			w = w - x[:,None];  #% subtract input vector
    #end
#end
#for i=1:num_pos_examples
	for i in range (num_pos_examples):
    #this_case = pos_examples(i,:);
		this_case = pos_examples[i,:]
    #x = this_case';
		x = this_case.T
    #activation = this_case*w;
		activation = asscalar(this_case.dot(w))
    #if (activation < 0)  % incorrect 0 output
		if activation < 0:  #% incorrect 0 output
        #%YOUR CODE HERE
			w = w + x[:,None];  #% add input vector
    #end
#end
	return w

#function [mistakes0, mistakes1] =  eval_perceptron(neg_examples, pos_examples, w)
def eval_perceptron(neg_examples, pos_examples, w):
	"""
%% 
% Evaluates the perceptron using a given weight vector. Here, evaluation
% refers to finding the data points that the perceptron incorrectly classifies.
% Inputs:
%   neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
%       num_neg_examples is the number of examples for the negative class.
%   pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
%       num_pos_examples is the number of examples for the positive class.
%   w - A 3-dimensional weight vector, the last element is the bias.
% Returns:
%   mistakes0 - A vector containing the indices of the negative examples that have been
%       incorrectly classified as positive.
%   mistakes1 - A vector containing the indices of the positive examples that have been
%       incorrectly classified as negative.
%%
	"""
#num_neg_examples = size(neg_examples,1);
	num_neg_examples = neg_examples.shape[0]
#num_pos_examples = size(pos_examples,1);
	num_pos_examples = pos_examples.shape[0]
#mistakes0 = [];
	mistakes0 = array([], dtype=int)
#mistakes1 = [];
	mistakes1 = array([], dtype=int)
	
#for i=1:num_neg_examples
	for i in range(num_neg_examples):
    #x = neg_examples(i,:)';
		x = neg_examples[i,:]
    #activation = x'*w;
		#print(x, w)
		activation = asscalar(x.dot(w))
    #if (activation >= 0)
		if activation >= 0:
        #mistakes0 = [mistakes0;i];
			mistakes0 =r_[mistakes0, i]
    #end
#end
#for i=1:num_pos_examples
	for i in range(num_pos_examples):
    #x = pos_examples(i,:)';
		x = pos_examples[i,:]
    #activation = x'*w;
		#print (w)
		activation = asscalar(x.dot(w).astype(float32))
    #if (activation < 0)
		if activation < 0:
        #mistakes1 = [mistakes1;i];
			mistakes1 = r_[mistakes1, i]
    #end
#end
	return mistakes0, mistakes1

def load_mat_file(filename):
	"""Load .mat file and return dictionary.
	
	Format:
	{'neg_examples_nobias':
		array([[-0.80857143,  0.8372093 ],
			[ 0.35714286,  0.85049834],
			[-0.75142857, -0.73089701],
			[-0.3       ,  0.12624585]]),
	'__header__':
		b'MATLAB 5.0 MAT-file, written by Octave 3.2.4, 2012-10-03 23:31:57 UTC',
	'w_gen_feas':
		array([[ 4.3496526 ],
			[-2.60997235],
			[-0.69414749]]),
	'__globals__':
			[],
	'__version__':
		'1.0',
	'w_init':
		array([[-0.62170147],
			[ 0.76091527],
			[ 0.77187205]]),
	'pos_examples_nobias':
		array([[ 0.87142857,
		0.62458472],
		[-0.02      , -0.92358804],
		[ 0.36285714, -0.31893688],
		[ 0.88857143, -0.87043189]])}	
	
	"""
	return scipy.io.loadmat('Datasets/' + filename)


if __name__ == '__main__':
	dataset = input('Enter number of dataset: ')
	data = load_mat_file('dataset'+ str(dataset) + '.mat')
	learn_perceptron(
		data['neg_examples_nobias'],
		data['pos_examples_nobias'],
		data['w_init'],
		data['w_gen_feas'])


