import random
import plr_perceptron as p

def testPerceptron(perceptron: p.Perceptron, truthTable: list, learningrate: int):
    '''
    
    '''
    trainingMax = 3
    initResults, initCorrect = perceptron.test(truthTable)
    i = 0
    corrects = 0
    while(i < len(initResults)):
        if(initResults[i] == initCorrect[i]):
            corrects += 1
        i += 1

    print(f'Running perceptron test. Starting with following values: \n'
          f'{perceptron} \n'
          f'Starting with the following truthTable {truthTable} \n'
          f'we get {corrects} out of {len(initResults)} correct before training. \n'
          f'Initializing up to {trainingMax} rounds of training...')
    perceptron.train(learningrate, truthTable, trainingMax)
    corrects = 0
    initResults, initCorrect = perceptron.test(truthTable)
    i = 0
    while(i < len(initResults)):
        if(initResults[i] == initCorrect[i]):
            corrects += 1
        i += 1
    print(f'We get {corrects} out of {len(initResults)} correct after {perceptron.RoundsTrained} rounds of training. \n'
          f'We obtained an RMSE value of: {perceptron.RMSE} \n'
          f'Our perceptron by the end has the following values: \n'
          f'{perceptron}')


random.seed(1797080)
learningrate = 0.1
andTable = [[0,0,0],[0,1,0],[1,0,0],[1,1,1]]
andPerceptron = p.Perceptron([random.random(),random.random()],random.random())
xorTable = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]
xorPerceptron = p.Perceptron([random.random(),random.random()],random.random())

# testPerceptron(andPerceptron, andTable, learningrate)
testPerceptron(xorPerceptron, xorTable, learningrate)
# flag = True
flag = False
while flag:
    print(f'Which section do you want to test? \n 1: And gate section \n 2: Xor gate Layer section \n 9: Quit')
    choice = int(input())
    if choice == 1:
        print(f'Testing training for the And gate:')
        testPerceptron(andPerceptron, andTable, learningrate)
    elif choice == 2:
        print(f'Testing training for the Xor gate:')
        testPerceptron(xorPerceptron, xorTable, learningrate)
    else:
        flag = False