import csv
import random
from Network import *
import math

############################################# Parameters to customize program

# Path to data CSV
csvPath = './data/iris_dataset.csv'

# Percent of data to be set aside as testing data
testingPercent = 20

# Number of neurons in hidden layer
numHiddenNeurons = 8

# Maximum number of epochs to complete
numEpochs = 10

# The smallest amount of error that is acceptable
maxError = 0.005

# The effect each epoch has on the weights
learningRate = 0.05

# Also effects how much epoch effects weights
momentum = 0.05

#############################################

def loadData():
    with open(csvPath) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        data = []
        for row in readCSV:
            # Read each value
            input1 = float(row[1])
            input2 = float(row[2])
            input3 = float(row[3])
            input4 = float(row[4])
            output1 = nameToNum(row[5])

            # Consolidate all inputs into one array
            temp = []
            temp.append(input1)
            temp.append(input2)
            temp.append(input3)
            temp.append(input4)
            temp.append(output1)

            # Append this set of data to set of all data
            data.append(temp)

        return data


def splitData():
    indices = getIndices(data)
    indices = shuffleIndices(indices)
    # Whatever isn't testing data will be training data
    trainingPercent = 100 - testingPercent

    # Multiply by decimal to get number of data
    numTraining = int(len(data) * trainingPercent / 100)
    numTesting = int(len(data) * testingPercent / 100)

    global trainingData
    trainingData = []
    global testingData
    testingData = []

    for i in range(numTraining):
        trainingData.append(data[i])

    for j in range(numTesting):
        # Pick up where trainging data ended
        testingData.append(data[j])



def getIndices(myList):
    # Gives list of numbers corresponding to indices
    indices = []
    for i in range(len(myList)):
        indices.append(i)
    return indices


def shuffleIndices(indices):
    # Uses the Fisher shuffling algorithm
    for i in range(len(indices)):
        randomNumber = random.randint(0, i)
        temp = indices[i];
        indices[i] = indices[randomNumber]
        indices[randomNumber] = temp
    return indices


def nameToNum(name):
    # Convert Iris name to number
    return {
        "setosa": 0,
        "versicolor": 1,
        "virginica": 2
    }[name]

def createNetwork():
    global neuralNetwork
    # Create network object
    # Hardcoded input and output because of data set
    neuralNetwork = Network(4, numHiddenNeurons, 3)

    # Initialize values for input layer
    for n in neuralNetwork.inputLayer:
        n.outgoingValue = 0
        for i in range(len(neuralNetwork.hiddenLayer)):
            randomNumber = random.randint(1, 10) / 1000
            n.outgoingWeights.append(randomNumber)
            n.outgoingChanges.append(0)

    # Initialize values for hidden layer
    for n in neuralNetwork.hiddenLayer:
        n.outgoingValue = 0
        for i in range(len(neuralNetwork.outputLayer)):
            randomNumber = random.randint(1, 10) / 1000
            n.outgoingWeights.append(randomNumber)
            n.outgoingChanges.append(0)

    # Initialize values for output layer
    for n in neuralNetwork.outputLayer:
        n.outgoingValue = 0
        n.outgoingWeights = None
        n.outgoingChanges = None


def train():
    indices = getIndices(trainingData)
    for i in range(numEpochs):
        neuralNetwork.printNetwork()
        # Shuffle indices
        indices = shuffleIndices(indices)
        # Make sure we're not overfitting the data by quitting before we get crazy accurate
        error = getGlobalError(trainingData)
        if error < maxError:
            break

        for j in indices:
            feedForward(trainingData[j])
            backProgogate(trainingData[j])


def getGlobalError(errData):
    error = 0
    for i in range(len(errData)):
        feedForward(errData[i])
        for j in range(len(neuralNetwork.outputLayer)):
            error = error + (errData[i][4] - neuralNetwork.outputLayer[j].outgoingValue) * (errData[i][4] - neuralNetwork.outputLayer[j].outgoingValue)
    return error / 3 * len(data)


def feedForward(datum):
    # Set input values to initial data
    for i in range(len(neuralNetwork.inputLayer)):
        neuralNetwork.inputLayer[i].outgoingValue = datum[i]

    # Compute outgoing values for hidden layer
    for i in range(len(neuralNetwork.hiddenLayer)):
        total = 0
        for j in range(len(neuralNetwork.inputLayer)):
            total = total + neuralNetwork.inputLayer[j].outgoingValue * neuralNetwork.inputLayer[j].outgoingWeights[i]
        activatedTotal = activationFunction(total)
        neuralNetwork.hiddenLayer[i].outgoingValue = activatedTotal

    # Compute outgoing values for output layer
    for i in range(len(neuralNetwork.outputLayer)):
        total = 0
        for j in range(len(neuralNetwork.hiddenLayer)):
            total = total + neuralNetwork.hiddenLayer[j].outgoingValue * neuralNetwork.hiddenLayer[j].outgoingWeights[i]
        activatedTotal = activationFunction(total)
        neuralNetwork.outputLayer[i].outgoingValue = activatedTotal


def activationFunction(num):
    return 1 / (1 + math.exp(-num))


def activationFunctionDerivative(num):
    return activationFunction(num) * (1 - activationFunction(num))


def backProgogate(datum):
    # Adjust weight in hidden layer based on the error in the output layer
    for i in range(len(neuralNetwork.outputLayer)):
        error = datum[4] - neuralNetwork.outputLayer[i].outgoingValue * activationFunctionDerivative(neuralNetwork.outputLayer[i].outgoingValue)
        for j in range(len(neuralNetwork.hiddenLayer)):
            weightChange = (error * learningRate * neuralNetwork.hiddenLayer[j].outgoingValue) + (momentum * neuralNetwork.hiddenLayer[j].outgoingChanges[i])
            neuralNetwork.hiddenLayer[j].outgoingWeights[i] = neuralNetwork.hiddenLayer[j].outgoingWeights[i] + weightChange
            neuralNetwork.hiddenLayer[j].outgoingChanges[i] = weightChange

    # Adjust weights in the input layer based on error in the output layer
    for i in range(len(neuralNetwork.hiddenLayer)):
        # Calculate error
        total = 0
        for j in range(len(neuralNetwork.outputLayer)):
            total = total + neuralNetwork.hiddenLayer[i].outgoingWeights[j] * (datum[4] - neuralNetwork.outputLayer[j].outgoingValue * activationFunctionDerivative(neuralNetwork.outputLayer[j].outgoingValue))
        error = total * activationFunctionDerivative(neuralNetwork.hiddenLayer[i].outgoingValue)

        # Adjust input layer weights
        for j in range(len(neuralNetwork.inputLayer)):
            weightChange = (error * learningRate * neuralNetwork.inputLayer[j].outgoingValue) + (momentum * neuralNetwork.inputLayer[j].outgoingChanges[i])
            neuralNetwork.inputLayer[j].outgoingWeights[i] = neuralNetwork.inputLayer[j].outgoingWeights[i] + weightChange
            neuralNetwork.inputLayer[j].outgoingChanges[i] = weightChange


def test(datum):
    count = 0
    for i in range(len(datum)):
        feedForward(datum[i])

        if isCorrect(datum[i]):
            count = count + 1

    return count / len(datum)


def isCorrect(datum):
    currentMaxValue = 0
    currentMaxPosition = 0

    # Identify the neuron outputting the highest confidence value
    for i in range(len(neuralNetwork.outputLayer)):
        if neuralNetwork.outputLayer[i].outgoingValue > currentMaxValue:
            currentMaxValue = neuralNetwork.outputLayer[i].outgoingValue
            currentMaxPosition = i

    return int(neuralNetwork.outputLayer[currentMaxPosition].outgoingValue) == datum[4]


if __name__ == "__main__":
    print()
    print("We're training a neural network!")
    print()

    global data
    data = loadData()
    splitData()
    createNetwork()
    train()

    trainingAccuracy = test(trainingData)
    testAccuracy = test(testingData)

    # Print accuracies, converting to percents
    print("Training accuracy: {}%".format(trainingAccuracy* 100))
    print("Testing accuracy: {}%".format(testAccuracy * 100))
