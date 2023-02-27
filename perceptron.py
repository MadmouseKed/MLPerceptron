"""
Perceptron

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
        


import itertools as it

class TestPerceptron:
    def assertEqual(x, y, z):
        if y == z:
            print('test success')
        else:
            print('test failed')
      
    def assertAny(x, y, z):
        for item in y:
            if item == z:
                print('test failed')
                break
            
        print('test success')
        
    
    def testINVERT(self):
        print('Running test INVERT')
        Invert = Perceptron([-1], 0)
        print('Testing with the following perceptron: {Invert}')
        print("Testing whether 1 -> 0")
        self.assertEqual(Invert.Activation([1]), 0)
        
        print("Testing whether 0 -> 1")
        self.assertEqual(Invert.Activation([0]), 1)
          
    def testAND(self):
        print('Running testAND')
        And = Perceptron([0.5, 0.5], -1)
        print(f'Testing with the following perceptron {And}')
        print("Testing whether for x1 =1 AND x2 =1 the result is 1")
        self.assertEqual(And.Activation([1,1]), 1)
        
        print('testing whether for x1 =0 ANd x2 =1 the result is 0')
        self.assertEqual(And.Activation([0,1]), 0)
        
        print('testing whether for x1 =1 AND x2 =0 the result is 0')
        self.assertEqual(And.Activation([1,0]), 0)
        
        print('testing whether for x1 =0 AND x2 =0 the result is 0')
        self.assertEqual(And.Activation([0,0]), 0)
        
    def testOR(self):
        print('Running testOR')
        Or = Perceptron([1, 1], -0.5)
        print(f'Testing with the following perceptron {Or}')
        print("Testing whether for x1 =1 AND x2 =1 the result is 1")
        self.assertEqual(Or.Activation([1,1]), 1)
        
        print('testing whether for x1 =0 ANd x2 =1 the result is 1')
        self.assertEqual(Or.Activation([0,1]), 1)
        
        print('testing whether for x1 =1 AND x2 =0 the result is 1')
        self.assertEqual(Or.Activation([1,0]), 1)
        
        print('testing whether for x1 =0 AND x2 =0 the result is 0')
        self.assertEqual(Or.Activation([0,0]), 0)
        
    def testNOR(self):
        print('Running testNOR')
        Nor = Perceptron([-1, -1, -1], 0)
        print(f'Testing with the following perceptron {Nor}')
        print("Testing whether for every combination of x1,x2 orx3 = 1, the result is 0")
        y = []
        for item in list(it.product([0,1], repeat = 3)):
            y.append(Nor.Activation(item))
        self.assertAny(y[1::], 1)
        print("Testing whether for x1,x2, and x3 = 0, the result is 1")
        self.assertEqual(Nor.Activation([0,0,0]), 1)
        
    def testNAND(self):
        print('Running testNAND')
        Nand = Perceptron([-0.6,-0.6], 1)
        print(f'Testing with the following perceptron {Nand}')
        print("Testing whether for every combination of x1,x2 orx3 = 1, the result is 0")
        y = []
        for item in list(it.product([0,1], repeat = 2)):
            activation = Nand.Activation(item)
            y.append(activation)
            print(f'x1,x2: {item} Net: {round(Nand.NetInput(item), 1)} Activation: {activation} ')

    def test2_8(self):
        print('Running test2_8')
        fig = Perceptron([0.6,0.3,0.2], -0.4)
        print(f'Testing with the following perceptron {fig}')
        x = list(it.product([0,1], repeat = 3))
        y = []
        print('Testing all combinations:')
        for item in x:
            activation = fig.Activation(item)
            print(f'x1,x2,x3: {item} Net: {round(fig.NetInput(item), 1)} Activation: {activation} ')
            y.append(activation)
        print('Testing that x2=1 or x3=1 on their own always results in 0')
        self.assertAny(y[:3:], 1)
        print('Testing leftover cases, x1=1 or x2=1 AND x3=1')
        self.assertAny(y[3::], 0)
        
    def testStraight(self):
        print('Running testStraight')
        straight = Perceptron([1], -0.1)
        print(f'Testing with the following perceptron {straight}')
        print('test taht x1=1 : 1')
        self.assertEqual(straight.Activation(1), 1)
        print('test that x1=0 : 0')
        self.assertEqual(straight.Activation(0), 0)
        
        
class TestPerceptronLayer:
    def assertEqual(x, y, z):
        if y == z:
            print('test success')
        else:
            
    
            print('test failed')
    
    def testLayer(self):
        layer = PerceptronLayer([Perceptron([-1,0,0],0), Perceptron([0,0.5,0.5], -1)])
        print(f'testing with the following PerceptronLayer: \n{layer}')
        print('testing whether with [0,1,1] we get [1,1] as result')
        self.assertEqual(layer.Activation([0,1,1]), [1,1])
    
class TestPerceptronNetwork:
    def assertEqual(x, y, z):
        if y == z:
            print('test success')
        else:
            print('test failed')

    def assertAny(x, y, z):
        for item in y:
            if item == z:
                print('test failed')
                break
            
        print('test success')

    def testXOR(self):
        layer1 = PerceptronLayer([Perceptron([1,0],-0.1),Perceptron([0,-1],0),Perceptron([-1,0],0),Perceptron([0,1],-0.1)])
        layer2 = PerceptronLayer([Perceptron([0.5,0.5,0,0],-1),Perceptron([0,0,0.5,0.5],-1)])
        layer3 = PerceptronLayer([Perceptron([1,1],-0.5)])
        network = PerceptronNetwork([layer1,layer2,layer3])
        print(f'testing with the following PerceptronNetwork: \n {network} ')
        print('Testing whether x1=0 and x2=0 or x1=1 and x2=1 results in 0')
        self.assertAny([network.Activation([0,0]),network.Activation([1,1])], [1])
        print('Testing whether x1=1 and x2=0 or x1=0 and x2=1 results in 1')
        self.assertAny([network.Activation([1,0]),network.Activation([0,1])], [0])
        
    def testAdder(self):
        layer1 = PerceptronLayer([Perceptron([1,0], -0.1),Perceptron([-1,-1], 0.5),Perceptron([0,1], -0.1)])
        layer2 = PerceptronLayer([Perceptron([-1,-1,0], 0.5),Perceptron([0,-1,-1], 0.5),Perceptron([-1,0,-1], 0.5)])
        layer3 = PerceptronLayer([Perceptron([-1,-1,0], 0.5),Perceptron([0,0,1], -0.1)])
        network = PerceptronNetwork([layer1,layer2,layer3])
        print(f'testing with the following PerceptronNetwork: \n {network} ')
        print('testing whether x1=0, x2=0 results in 1, 1')
        self.assertEqual(network.Activation([0,0]), [1,1])
        print('testing whether x1=0, x2=1 results in 1, 0')
        self.assertEqual(network.Activation([0,1]), [0,0])
        print('testing whether x1=1, x2=0 results in 0, 0')
        self.assertEqual(network.Activation([1,0]), [0,0])
        print('testing whether x1=1, x2=1 results in 0, 1')
        self.assertEqual(network.Activation([1,1]), [1,0])
        
def tests1():
    print("##########")
    print('Perceptron section')
    tests1 = TestPerceptron()
    tests1.testINVERT()
    print("---------")
    tests1.testAND()
    print("---------")
    tests1.testOR()
    print("---------")
    tests1.testNOR()
    print("---------")
    tests1.test2_8()
    print("---------")
    tests1.testNAND()
    print("---------")
    tests1.testStraight()
    print(' ')

def tests2():
    print("##########")
    print('Percepton Layer section')
    tests2 = TestPerceptronLayer()
    print("---------")
    tests2.testLayer()
    print(' ')

def tests3():
    print("#########")
    print('Percepton Network Section')
    tests3 = TestPerceptronNetwork()
    print("---------")
    tests3.testXOR()
    print("---------")
    tests3.testAdder()
    print("---------")


flag = True
while flag:
    print(f'Which section do you want to test? \n 1: Perceptron section \n 2: Perceptron Layer section \n 3: Perceptron Network section \n 4: Quit')
    choice = int(input())
    if choice == 1:
        tests1()
    elif choice == 2:
        tests2()
    elif choice == 3:
        tests3()
    else:
        flag = False
