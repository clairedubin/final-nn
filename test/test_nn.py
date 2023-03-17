# TODO: import dependencies and write unit tests below
import pytest
import numpy as np
from nn.preprocess import *
from nn.nn import NeuralNetwork
from sklearn.metrics import log_loss, mean_squared_error
import warnings

def load_example_NN():

    return NeuralNetwork(nn_arch = [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'},
                                    {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}],
                          lr = 0.01, seed = 3, batch_size = 100, 
                          epochs = 1, loss_function='log loss')



def test_single_forward():

    example_NN = NeuralNetwork(nn_arch = [{'input_dim': 10, 'output_dim': 32, 'activation': 'relu'}],
                          lr = 0.01, seed = 3, batch_size = 100, 
                          epochs = 10, loss_function='log loss')

    X = np.random.rand(100, 10)
    weights = example_NN._param_dict['W1']
    bias = example_NN._param_dict['b1']

    #check that unknown activation layer causes error
    with pytest.raises(Exception) as exc_info:
        example_NN._single_forward(weights, bias, X, 'this is not an activation')
    assert str(exc_info.value) == 'activation function this is not an activation not supported' 

    #check that dimensions of A and Z are as expected for relu
    A_out, Z_out = example_NN._single_forward(weights, bias, X, 'relu')
    assert A_out.shape == (100, 32), 'A dimensions are not as expected (input_dim, output_dim)'
    assert Z_out.shape == (100, 32), 'Z dimensions are not as expected (input_dim, output_dim)'
    
    #check that dimensions of A and Z are as expected for sigmoid (same as relu)
    A_out, Z_out = example_NN._single_forward(weights, bias, X, 'sigmoid')
    assert A_out.shape == (100, 32), 'A dimensions are not as expected (input_dim, output_dim)'
    assert Z_out.shape == (100, 32), 'Z dimensions are not as expected (input_dim, output_dim)'

def test_forward():
    
    num_samples = 100
    input_dim = 10
    final_output_dim = 64

    example_NN = NeuralNetwork(nn_arch = [{'input_dim': input_dim, 'output_dim': 32, 'activation': 'relu'},
                                          {'input_dim': 32, 'output_dim': final_output_dim, 'activation': 'relu'}],
                          lr = 0.01, seed = 3, batch_size = 100, 
                          epochs = 10, loss_function='log loss')

    X = np.random.rand(num_samples, input_dim)
    output, cache = example_NN.forward(X)

    #check that output layer has dimensions (num_samples, final_output_dim)
    assert output.shape == (num_samples, final_output_dim), "Outut of last layer does not have expected dimensions"
    
    #check that cache has data for each layer + input layer)
    assert len(cache) == len(example_NN.arch) + 1, "Cache outputs not equal to number of layers + 1"
    
    #check that cache changes with new inputs
    new_X = np.random.rand(num_samples, input_dim)
    output, new_cache = example_NN.forward(new_X)
    for layer in cache:
        for element, matrix in cache[layer].items():
            assert not np.array_equal(matrix, new_cache[layer][element]), \
            '{} caches for layer {} are equal across runs with different inputs'.format(element,layer)

def test_single_backprop():

    num_samples = 5
    input_dim = 10
    output_dim = 2

    example_NN = NeuralNetwork(nn_arch = [{'input_dim': input_dim, 'output_dim': output_dim, 'activation': 'relu'},
                                          ],
                          lr = 0.01, seed = 3, batch_size = 100, 
                          epochs = 10, loss_function='log loss')

    X = np.random.rand(num_samples, input_dim)
    y = np.random.rand(num_samples, output_dim) 

    output, cache = example_NN.forward(X)
    loss_backprop = example_NN._binary_cross_entropy_backprop(y, output)
    weights = example_NN._param_dict['W1']
    bias = example_NN._param_dict['b1']

    #check that unknown activation function causes error
    with pytest.raises(Exception) as exc_info:
        example_NN._single_backprop(weights, bias, output, X, loss_backprop, 'this is not an activation')
    assert str(exc_info.value) == 'activation function this is not an activation not supported' 

    #check that shapes are correct
    dA, dW, db = example_NN._single_backprop(weights, bias, output, X, loss_backprop, 'relu')
    assert dA.shape == X.shape, "dA shape not equal to A_prev shape"
    assert dW.shape == weights.shape, "dW shape not equal to shape of weights"
    assert db.shape == bias.shape, "dB shape not equal to shape of bias"


def test_predict():

    input_dim = 10
    output_dim = 2

    example_NN = NeuralNetwork(nn_arch = [{'input_dim': input_dim, 'output_dim': output_dim, 
                                           'activation': 'sigmoid'}],
                          lr = 0.01, seed = 3, batch_size = 100, 
                          epochs = 10, loss_function='mse')
    
    X_train = np.random.rand(50, input_dim)
    X_val = np.random.rand(10, input_dim)
    y_train = np.random.rand(50, output_dim) 
    y_val = np.random.rand(10, output_dim) 
    X_test = np.random.rand(20, input_dim)
    y_test = np.random.rand(20, output_dim)

    #check that warning is given when predict is run before fit
    with pytest.warns(UserWarning) as w:
        example_NN.predict(X_test)

    #check that warning doesn't show after fit
    example_NN.fit(X_train, y_train, X_val, y_val)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        example_NN.predict(X_test)
    
    #check that predict and fit give different outputs for various combinations of loss and activation functions
    preds = []
    for loss in ['log loss', 'mse']:
        for activation in ['sigmoid', 'relu']:
            example_NN = NeuralNetwork(nn_arch = [{'input_dim': input_dim, 
                                                   'output_dim': output_dim, 
                                                'activation': activation}],
                                lr = 0.01, seed = 3, batch_size = 100, 
                                epochs = 10, loss_function=loss)

            example_NN.fit(X_train, y_train, X_val, y_val)
            preds += [example_NN.predict(X_test)]

    for i, pred1 in enumerate(preds):
        for j, pred2 in enumerate(preds):
            if i == j:
                continue
            assert not np.array_equal(pred1, pred2)

def test_binary_cross_entropy():

    #check that BCE of example data matches value calculated by sklearn
    example_NN = load_example_NN()
    y = np.array([0,0,0,1,1,1])
    y_hat = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 1])
    correct_BCE = log_loss(y, y_hat)
    output_BCE = example_NN._binary_cross_entropy(y, y_hat)
    assert np.isclose(output_BCE, correct_BCE), 'MSE does not match expected value'

def test_binary_cross_entropy_backprop():
    
    example_NN = load_example_NN()
    y = np.array([0,1,2])
    y_hat = np.array([3,2,-4])

    #check that output value matches value calculated by hand
    correct_BCE_backprop = np.array([-0.166666,-0.166666, 0.1])
    output_BCE_backprop = example_NN._binary_cross_entropy_backprop(y, y_hat)
    assert np.allclose(output_BCE_backprop, correct_BCE_backprop), \
    'BCE derivative does not match expected value'

def test_mean_squared_error():

    #check that MSE of example data matches value calculated by sklearn
    example_NN = load_example_NN()
    y = np.array([0,1,2,3,4,5])
    y_hat = np.array([-2,1,-1,3,8,5])
    correct_MSE = mean_squared_error(y,y_hat)
    output_MSE = example_NN._mean_squared_error(y, y_hat)
    assert np.isclose(output_MSE, correct_MSE), 'MSE does not match expected value'

def test_mean_squared_error_backprop():
    
    example_NN = load_example_NN()
    y = np.array([0,1,2])
    y_hat = np.array([1,2,-4])

    #check that output value matches value calculated by hand
    correct_MSE_backprop = np.array([0.666666,0.666666,-4])
    output_MSE_backprop = example_NN._mean_squared_error_backprop(y, y_hat)
    assert np.allclose(output_MSE_backprop, correct_MSE_backprop), \
    'MSE derivative does not match expected value'

def test_sample_seqs():

    #check that sequence types and labels are balanced after function is run

    seqs = ['ATGC', 'ATGTTA', 'MFGRS', 'MDKFL', 'MRRKF', 'MFQHD', 'MFHIR']
    labels = ['DNA', 'DNA', 'protein', 'protein', 'protein', 'protein', 'protein']
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)

    DNA_seqs = [i for i,seq in enumerate(sampled_seqs) if seq[:3] == 'ATG']
    protein_seqs =  [i for i,seq in enumerate(sampled_seqs) if seq[0] == 'M']

    assert len(DNA_seqs) == len(protein_seqs), "Sequences are still uneven after sampling"

    for i in DNA_seqs:
        assert sampled_labels[i] == 'DNA', "Sample label doesn't match DNA sequence"
    for i in protein_seqs:
        assert sampled_labels[i] == 'protein', "Sample label doesn't match protein sequence"

def test_one_hot_encode_seqs():

    #check that a sequence is correctly one-hot encoded
    test_seq = ['GATC']
    correct_one_hot = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])

    one_hot_out = one_hot_encode_seqs(test_seq)[0]
    assert np.array_equal(one_hot_out, correct_one_hot), "One hot encoding function output doesn't match predicted output"