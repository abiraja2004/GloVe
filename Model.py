'''
Hyperparameters :   d               -> dimension of the word_vector. 
                    alpha           -> a parameter of the weighing factor {Refer to the GloVe paper (GloVe.pdf)}
                    x_max           -> a parameter of the weighing factor {Refer to the GloVe paper (GloVe.pdf)}
                    learning_rate   -> learning rate used with AdaGrad
'''
import numpy as np


#Path where coccurrence matrix is stored
COCCURRENCE_MATRIX_PATH = "/Data/Coocuurence matrix News 2.npy"
OUTPUT_DIR = '/output/'

'''
Initialises the parameters for both word vectors and the square of gradients required for AdaGrad optimization 

parameters: V       -> size of the vocabulary
            d       -> dimension of the word vector
            
returns   : params  -> Dictionary of U, V, b and c vectors
                       U -> Vectors for center/main words
                       V -> Vectors for context words
                       b, c -> biases {Refer to the GloVe paper (GloVe.pdf)}
            grad_sq -> Maintains the sum of the square of gradients for each of the above parameters.
                       Required for AdaGrad update 
'''
def initalise_parameters(V, d):
    params = {}
    grad_sq = {}
    params["U"] = np.random.randn(V, d) - 0.5
    params["V"] = np.random.randn(V, d) - 0.5
    params["b"] = np.random.randn(V) - 0.5
    params["c"] = np.random.randn(V) - 0.5
    grad_sq["U"] = np.zeros_like(params["U"]) 
    grad_sq["V"] = np.zeros_like(params["V"]) 
    grad_sq["b"] = np.zeros_like(params["b"]) 
    grad_sq["c"] = np.zeros_like(params["c"]) 
    return params, grad_sq

'''
Load saved parameters. 
'''
def use_trained_parameters():
    params = {}
    grad_sq = {}
    params["U"] = np.load('./Param_U9.npy')
    params["V"] = np.load('./Param_V9.npy')
    params["b"] = np.load('./Param_b9.npy')
    params["c"] = np.load('./Param_c9.npy')
    grad_sq["U"] = np.load('./grad_sq_U9.npy')
    grad_sq["V"] = np.load('./grad_sq_V9.npy')
    grad_sq["b"] = np.load('./grad_sq_b9.npy')
    grad_sq["c"] = np.load('./grad_sq_c9.npy')
    return params, grad_sq

'''
The weighing factor {Refer to the GloVe paper (GloVe.pdf)}
'''
def f(x, x_max = 100, alpha = 0.75):
    if x > x_max:
        return 1
    else:
        return (x/x_max)**(alpha)

'''
Performs one pass through the cooccurence matrix and updates the parameters using AdaGrad rule

parameters: X               -> size of the vocabulary
            vocab_size      -> dimension of the word vector
            params          -> Dictionary of U, V, b and c vectors
                               U -> Vectors for center/main words
                               V -> Vectors for context words
                               b, c -> biases {Refer to the GloVe paper (GloVe.pdf)}
            grad_sq         -> sum of the square of gradients for each of the above parameters
            learning_rate   -> Used for AdaGrad update
            epsilon         -> fudge factor, to avoid divide by zero {Refer to the AdaGrad paper}
            
returns   : params          -> updated params
            grad_sq         -> updated grad_sq 
'''
def update_parameters(X, vocab_size, params, grad_sq, learning_rate, epsilon=10**(-6)):
    cost = 0
    perm1 = np.random.permutation(vocab_size)
    perm2 = np.random.permutation(vocab_size)
    for i in perm1:
        for j in perm2:
            f_x = f(X[i, j]) 
            if f_x > 0:
                prod = f_x * (np.sum(params["U"][i]  * params["V"][j]) + params["b"][i] + params["c"][j] - np.log(X[i, j]))
                cost = cost + 0.5 * prod * (np.sum(params["U"][i]  * params["V"][j]) + params["b"][i] + params["c"][j] - np.log(X[i, j]))
                dUi = prod*params["V"][j]
                dVj = prod*params["U"][i]
                dbi = prod
                dcj = prod
                
                grad_sq["U"][i] += np.square(dUi)
                grad_sq["V"][j] += np.square(dVj)
                grad_sq["b"][i] += dbi**2
                grad_sq["c"][j] += dcj**2
                
                params["U"][i] = params["U"][i] - learning_rate * dUi/(np.sqrt(grad_sq["U"][i]) + epsilon)
                params["V"][j] = params["V"][j] - learning_rate * dVj/(np.sqrt(grad_sq["V"][j]) + epsilon)
                params["b"][i] = params["b"][i] - learning_rate * dbi/(np.sqrt(grad_sq["b"][i]) + epsilon)
                params["c"][j] = params["c"][j] - learning_rate * dcj/(np.sqrt(grad_sq["c"][j]) + epsilon)
    print('{"metric": "Cost", "value": ' + str(cost) + ' }') # An estimate of the cost for the pass
    return params, grad_sq

'''
Train for given number of epochs with specified learning rate.
Saves the output in the directory specified

parameters: epochs          -> number of epochs to train
            learning_rate   -> used for AdaGrad update
            d               -> dimension of the word vectors
'''
def main(epochs=5, learning_rate=0.05, d=50):
    X = np.load(COCCURRENCE_MATRIX_PATH)  
    #X = np.load("../Data/Coocuurence matrix News 2.npy")
    
    vocab_size = X.shape[0]
    
    #params, grad_sq = initalise_parameters(vocab_size, d)
    params, grad_sq = use_trained_parameters()
    assert(params["U"].shape[1] == d)
    
    for i in range(epochs):
        params, grad_sq = update_parameters(X, vocab_size, params, grad_sq)
        if i % 5 == 4:
            np.save(OUTPUT_DIR+'Param_U'+str(i), params["U"])
            np.save(OUTPUT_DIR+'Param_V'+str(i), params["V"])
            np.save(OUTPUT_DIR+'Param_b'+str(i), params["b"])
            np.save(OUTPUT_DIR+'Param_c'+str(i), params["c"])
            np.save(OUTPUT_DIR+'grad_sq_U'+str(i), grad_sq["U"])
            np.save(OUTPUT_DIR+'grad_sq_V'+str(i), grad_sq["V"])
            np.save(OUTPUT_DIR+'grad_sq_b'+str(i), grad_sq["b"])
            np.save(OUTPUT_DIR+'grad_sq_c'+str(i), grad_sq["c"])

main(10, 0.05, 50)
