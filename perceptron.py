"""
Perceptron by W.Ketel 2023
"""

import numpy as np

class Perceptron(object):
    ''' 
    A perceptron class for individual perceptrons
    Param weight: A list of float value weights, how strong the perceptron favours an input given on that channel.
    Param bias: The activation bias, how likely the Perceptron is to activate.
    '''
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __str__(self):
        '''
        Shows what parts the perceptron is made out of.
        '''
        return f'Weight: {self.weight} Bias: {self.bias}'
    
    def NetInput(self, inputs): #w-> * x->
        '''
        A supporting function to calculate the numerical value of the input.
        Param self.weights: The list of weights used to initialize the perceptron
        Param inputs
        returns: A numerical value, The dot product of self.weights with inputs. 
        ''' 
        NetSum = np.dot(self.weight, inputs)
        return NetSum

        
    def Activation(self, inputs): #g=0 if: w-> * x-> +b < 0 else g=1 if: w-> * x-> +b >= 0
        '''
        Determins whether the perceptron is going to activate or not.
        Param self.bias: The offset for how likely the perceptron is to activate or not.
        Param inputs: A list of input values.
        returns: a 0 or 1, depending on whether the activation treshhold has been met or not.
        '''
        NetSum = self.NetInput(inputs)
        if NetSum + self.bias < 0:
            return 0
        elif NetSum + self.bias >= 0:
            return 1

class PerceptronLayer(Perceptron):
    '''
    A perceptron clas for an individual layer of perceptrons.
    Inherit: Perceptron, as we need to know what perceptrons consist off.
    Param perceptron: The list of Perceptrons that make up the layer.
    '''
    def __init__(self, perceptron):
        self.perceptron = perceptron
        
    def __str__(self):
        '''
        Returns information on each Perceptron that makes up the layer.
        '''
        text = 'This layer contains the following Perceptrons: \n'
        for i in self.perceptron:
            text += str(i) + '\n'
        return text
    
    def Activation(self, inputs):
        '''
        A function to build a list of which Perceptrons activate and which don't based on inputs list.
        We check in order, so if we have 3 perceptrons and 3 inputs, we assume that the 1st input is for the 1st perceptron.
        param self.perceptron: the list of Perceptrons which we'll check for activation or not.
        param inputs: the list of values that will be checked against the Perceptron input.
        returns: A list of results, of whether each individual Perceptron activated, this list is ordered.
        '''
        results = []
        for i in self.perceptron:
            results.append(i.Activation(inputs))
        return results
            
class PerceptronNetwork(PerceptronLayer):
    '''
    A Perceptron class for an individual Network made up of multiple Perceptron layers.
    Inherit: PerceptronLayer, because we need to know what a layer is made out of.
    Param perceptronLayers: the list of PerceptronLayers which make up the network. 
    We assume that each layer is sequential, so layer1 will feed into layer2, and layer2 will feed into layer3 etc.
    '''
    def __init__(self, perceptronLayers):
        self.layers = perceptronLayers
    
    def __str__(self):
        '''
        Returns information on each PerceptronLayer that makes up the network.
        '''
        text = 'this Network consists of the following layers: \n'
        for i in self.layers:
            text += str(i)
        return text

    def Activation(self, inputs):
        '''
        Function to build a list on what the ultimate outcome is going to be of the network, dependant on its inputs
        param self.layers: the list of layers which make up the network.
        param inputs: a list of inputs, note that this should be scaled for the lowest order Layer
        returns: The list of results.
        '''
        result = inputs
        for layer in self.layers:
            result = layer.Activation(result)
        return result