import math

class Neuron(object):
    ''' 
    A Neuron class for individual Neurons
    Param weight: A list of float value weights, how strong the Neuron favours an input given on that channel.
    Param bias: The activation bias, an offset value for how much the Neuron activates.
    '''
    def __init__(self, weight : list, bias : float):
        self.weight = weight
        self.bias = bias

    def __str__(self):
        '''
        Shows what the neuron is made off
        '''
        return f'Weight: {self.weight} Bias: {self.bias}'
    
    
    def NetInput(self, input : list):
        '''
        A supporting function to calculate the numerical float value of the input.
        Param self.weights: The list of weights used to initialize the Neuron.
        Param inputs: The list of input values used to activate the Neuron
        returns: A numerical value, The dot product of self.weights with inputs. 
        ''' 
        i = 0
        NetSum = 0
        while(i < len(input)):
            NetSum += (self.weight[i]*input[i] - self.bias)
            i += 1
        return NetSum


    def Activation(self, input : list) -> float:
        '''
        Calculates the output value of the Neuron based on the inputs provided.
        Param inputs: A list of input values.
        returns: a float value.
        '''
        return 1/(1 + math.exp(-self.NetInput(input)))

class NeuronLayer(Neuron):
    '''
    A Neuron clas for an individual layer of Neurons.
    Inherit: Neuron, as we need to know what Neurons consist off.
    Param neurons: The list of Neurons that make up the layer.
    '''
    def __init__(self, neurons : list):
        self.Neurons = neurons

    def __str__(self):
        '''
        Returns information on each Neuron that makes up the layer.
        '''
        text = 'This layer contains the following Perceptrons: \n'
        for i in self.Neurons:
            text += str(i) + '\n'
        return text
    
    def Activation(self, inputs):
        '''
        A function to build a list of which Neurons activate and to what extend based on inputs list.
        We check in order, so if we have 3 Neurons and 3 inputs, we assume that the 1st input is for the 1st Neuron.
        param self.Neuron: the list of Neurons which we'll check for activation or not.
        param inputs: the list of values that will be checked against the Neuron input.
        returns: A list of results, of  Neuron activation values, this list is ordered.
        '''
        results = []
        for i in self.Neurons:
            results.append(i.Activation(inputs))
        return results

class NeuronNetwork(NeuronLayer):
    '''
    A Neuron class for an individual Network made up of multiple NeuronLayers.
    Inherit: NeuronLayer, because we need to know what a layer is made out of.
    Param neuronLayers: the list of NeuronLayers which make up the network. 
    We assume that each layer is sequential, so layer1 will feed into layer2, and layer2 will feed into layer3 etc.
    '''
    def __init__(self, neuronLayers : list):
        self.layers = neuronLayers

    def __str__(self):
        '''
        Returns information on each NeuronLayer that makes up the network.
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

# layer1 = sig.NeuronLayer([sig.Neuron([40,0], 20),Nand,sig.Neuron([0,40], 20)])
# layer2 = sig.NeuronLayer([sig.Neuron([100,100,0], 80),sig.Neuron([0,100,100], 80),sig.Neuron([13,0,13], 10)])
# layer3 = sig.NeuronLayer([sig.Neuron([100,100,0], 80),sig.Neuron([0,0,40], 20)])
# test = Neuron([40,0], 20)
# test = Neuron([100,100,0], 80)
# test = Neuron([-200,-200,0], -80)
test = Neuron([30,0,30], 10)
print(round(test.Activation([1,0,1]),5))