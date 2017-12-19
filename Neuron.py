class Neuron:
    def __init__(self):
        self.outgoingValue = None
        self.outgoingWeights = []
        self.outgoingChanges = []

    def printNeuron(self):
        print(self.outgoingValue)
        print(self.outgoingWeights)
        print(self.outgoingChanges)



if __name__ == "__main__":
    n = Neuron(1, 0, 0)
    n.printNeuron()
