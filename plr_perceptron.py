"""
Perceptron Learning rule
W.Ketel 2023
"""
class Perceptron(object):
    ''' 
    A perceptron class for individual perceptrons.
    Param weight: A list of float value weights, how strong the perceptron favours an input given on that channel.
    Param bias: The activation bias, how likely the Perceptron is to activate overal.
    '''
    def __init__(self, weight: list, bias: float):
        self.weight = weight
        self.bias = bias
        self.RMSE = 0
        self.RoundsTrained = 0 #This one is superfluous, we don't actually need it, but I thought it might be nice to see how well the alghoritm performs.

    def __str__(self):
        '''
        Shows what parts the perceptron is made out of.
        '''
        return f'Weight: {self.weight} Bias: {self.bias}'
    
    def update(self, target: int, input: list, learningrate: float): #target d, input x->, weights w->, bias b
        '''
        Utilizes the perceptron learning rule: \Delta w_{j} = \eta(target^{(i)} - output^{(i)})x_{j}^{(i)}
        Param target: the desired outcome
        Param input: The list of inputs given.
        Param learningrate: the learningrate to be used. 
        '''
        error = target - self.Activation(input) #e = d - f(w->*x->) {0,1}
        deltaW = list(map(lambda x:x*error*learningrate, input))
        deltaB = learningrate * error
        wAccent = [sum(x) for x in zip(*[self.weight, deltaW])]
        bAccent = self.bias + deltaB
        self.weight = wAccent
        self.bias = bAccent


    def loss(self, number: float, count: int) -> None: #target d, output y, training examples n
        '''
        Calculating the Mean Square Error of the training run. And updates self.RMSE to be the correct value
        Param number: The pre-agregated result of sum |d - y|^2
        Param count: How many training cycles were performed upon the Perceptron.
        '''
        self.RMSE = number / count

    def test(self, input : list) -> 'tuple[list, list]':
        '''
        Builds a results and correct list for usage in train function.
        Param input: list of inputs.
        '''
        results = []
        correct = []
        for i in input:
            results.append(self.Activation(i[:-1]))
            correct.append(i[-1])
        return results, correct


    def train(self, learningrate: float, truthtable: list, itercount: int) -> None:
        '''
        Training function for the perceptron. Using the learning rate and the truthtable, it will train the perceptron untill it manages to give the correct target result based on the input lists.
        Param learningrate: a float value between 0 and 1.0
        Param truthtable: A table with the correct outcomes. Made up from a shell list containing lists, each list contains: Input1, input2, input3.., target
        '''
        results, correct = self.test(truthtable)
        i = 0
        num = 0 #sum abs(d - y)^2 / i
        while((results != correct) and (i < itercount)):
            for category in truthtable:
                if(category[-1] != self.Activation(category[:-1])):
                    self.update(category[-1],category[:-1],learningrate)
                    num += (category[-1] - self.Activation(category[:-1]))^2
            results, correct = self.test(truthtable)
            i += 1
        self.loss(num, i)
        self.RoundsTrained = i


    def NetInput(self, inputs: list) -> float: #w-> * x->
        '''
        A supporting function to calculate the dotproduct value of the input with the perceptrons weights.
        Param self.weights: The list of weights used to initialize the perceptron
        Param inputs: The list of inputs given.
        returns: A numerical value, The dot product of self.weights with inputs. 
        ''' 
        NetSum = 0
        i = 0
        while i < len(inputs):
            NetSum += self.weight[i] * inputs[i]
            i += 1
        return NetSum

        
    def Activation(self, inputs: list) -> int: #g=0 if: w-> * x-> +b < 0 else g=1 if: w-> * x-> +b >= 0
        '''
        Determins whether the perceptron is going to activate or not.
        Param self.bias: The offset for how likely the perceptron is to achieve activation.
        Param inputs: A list of input values.
        returns: a 0 or 1, depending on whether the activation treshhold has been met or not.
        '''
        NetSum = self.NetInput(inputs)
        if NetSum + self.bias < 0:
            return 0
        elif NetSum + self.bias >= 0:
            return 1
