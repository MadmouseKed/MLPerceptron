import sigmoid_neuron as sig

class testSigmoidNeuron:
    ### The problem with initalizing the sigmoid neuron with the same values as a Neuron:
    ### For the Neuron we used a cut off, any value 0.5 or higher became 1, any value smaller than 0.5 became 0, therefore a 0.27 resulted in 0 and 0.5 in 1,
    ### sigmoid does not work this way, if in Activation() it calculates the result to be 0.27, then that is what will be returned, therefore we have to adjust the weights and biases of each
    ### Neuron so that the intended behaviour is preserved.
    def testINVERT(self):
        ### We want to retain  1 -> 0 and 0 -> 1,
        ### 1/ x requires x to as close to 1, if we want an output of 1, and x has to be a very large number, if we want it to result in 0.
        ### For our function, this means that for 0 -> 1 we'd like e^(-sum(w->*x-> -b)) to be about 0, since then we get 1/1 = 1, e(-10) results in about 0, 
        ### so...- (w*x - b) = -10, x = 0, so w*x =0, therefore b has to be -10 (---10 = -10)
        ### then for 1 -> 0, we want e(y) to be as big as possible, e(90) results in about 2e5, which is plenty sufficient, and we reach that by putting the weight at -100.
        print('Test Invert')
        Invert = sig.Neuron([-100], -10)
        print(f'Testing with the following Neuron: {Invert} Rounding result to 5 digits.')
        print(f'Testing whether 1 -> 0: {round(Invert.Activation([1]),5)}')
        print(f'Testing whether 0 -> 1: {round(Invert.Activation([0]),5)}')

    def testAND(self):
        ### We want to retain 1,1 -> 1 : 0,1 -> 0 : 1,0 -> 0 : 0,0 -> 0
        ### We already discussed in testINVERT how we go about acquiring the desired results, but this time we actually have to deal with the summation part of the equation.
        ### For this we know that the -b, applies every single round. We also know that our inputs are pretty straightforwards. We will be giving the system 1 or 0, so we can still make some
        ### safe assumptions about how to achieve our desired outcomes.
        ### x1 and x2, are both equally important, so we can say x1 = x2, after all with the AND gate we want both of them to be present in order to activate.
        ### We also need to make sure bias is signifciant enough that only by summing both x1 and x2's results, that a good value is achieved. For all the outcomes with 0, we want e(big)
        ### 
        print('Running testAND')
        And = sig.Neuron([13, 13], 10)
        print(f'Testing with the following Neuron {And}')
        print(f"Testing whether for x1 =1 AND x2 =1 the result is 1: {round(And.Activation([1,1]),5)}")
        print(f'testing whether for x1 =0 ANd x2 =1 the result is 0: {round(And.Activation([0,1]),5)}')
        print(f'testing whether for x1 =1 AND x2 =0 the result is 0: {round(And.Activation([1,0]),5)}')
        print(f'testing whether for x1 =0 AND x2 =0 the result is 0: {round(And.Activation([0,0]),5)}')

    def testOR(self):
        print('Running testOR')
        Or = sig.Neuron([100, 100], 10)
        print(f'Testing with the following Neuron {Or}')
        print(f"Testing whether for x1 =1 AND x2 =1 the result is 1: {round(Or.Activation([1,1]),5)}")
        print(f'testing whether for x1 =0 ANd x2 =1 the result is 1: {round(Or.Activation([0,1]),5)}')
        print(f'testing whether for x1 =1 AND x2 =0 the result is 1: {round(Or.Activation([1,0]),5)}')
        print(f'testing whether for x1 =0 AND x2 =0 the result is 0: {round(Or.Activation([0,0]),5)}')

    def testNOR(self):
        print('Running testNOR')
        Nor = sig.Neuron([-100, -100, -100], -10)
        print(f'Testing with the following Neuron {Nor}')
        print(f'Testing for x1 =0, x2 =0, x3 = 0, output = 1: {round(Nor.Activation([0,0,0]),5)}')
        print(f'Testing for x1 =1, x2 =0, x3 = 0, output = 0: {round(Nor.Activation([1,0,0]),5)}')
        print(f'Testing for x1 =1, x2 =1, x3 = 1, output = 0: {round(Nor.Activation([1,1,1]),5)}')

    def testNAND(self):
        print('Running testNAND')
        Nand = sig.Neuron([-100,-100], -80)
        print(f'testing with teh following Neuron {Nand}')
        print(f'Testing for x1 =0, x2 =0, output = 1: {round(Nand.Activation([0,0]),5)}')
        print(f'Testing for x1 =0, x2 =1, output = 1: {round(Nand.Activation([0,1]),5)}')
        print(f'Testing for x1 =1, x2 =0, output = 1: {round(Nand.Activation([1,0]),5)}')
        print(f'Testing for x1 =1, x2 =1, output = 0: {round(Nand.Activation([1,1]),5)}')


class testSigmoidNeuronNetwork:
    def testAdder(self):
        # straight = sig.Neuron([40], 20)
        # Nand = sig.Neuron([-100,-100], -80)
        # And = sig.Neuron([13, 13], 10)
        layer1 = sig.NeuronLayer([sig.Neuron([40,0], 20),Nand,sig.Neuron([0,40], 20)])
        layer2 = sig.NeuronLayer([sig.Neuron([-200,-200,0], -80),sig.Neuron([0,-200,-200], -80),sig.Neuron([30,0,30], 10)])
        layer3 = sig.NeuronLayer([sig.Neuron([-200,-200,0], -80),sig.Neuron([0,0,100], 10)])
        network = sig.NeuronNetwork([layer1,layer2,layer3])
        print(f'testing with the following PerceptronNetwork: \n {network} ')
        print(f'testing whether x1=0, x2=0 results in [0, 0]: {network.Activation([0,0])}')
        print(f'testing whether x1=0, x2=1 results in [1, 0]: {network.Activation([0,1])}')
        print(f'testing whether x1=1, x2=0 results in [1, 0]: {network.Activation([1,0])}')
        print(f'testing whether x1=1, x2=1 results in [0, 1]: {network.Activation([1,1])}')

def tests1():
    print("##########")
    print('Neuron section')
    tests1 = testSigmoidNeuron()
    tests1.testINVERT()
    print("---------")
    tests1.testAND()
    print("---------")
    tests1.testOR()
    print("---------")
    tests1.testNOR()
    print("---------")
    tests1.testNAND()

def tests2():
    print("#########")
    print('Percepton Network Section')
    tests2 = testSigmoidNeuronNetwork()
    print("---------")
    tests2.testAdder()
    print("---------")

tests1()
tests2()

flag = True
while flag:
    print(f'Which section do you want to test? \n 1: Neuron section \n 2: NeuronLayer section \n else: Quit')
    choice = int(input())
    if choice == 1:
        tests1()
    elif choice == 2:
        tests2()
    else:
        flag = False