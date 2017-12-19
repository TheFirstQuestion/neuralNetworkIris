from Neuron import *

class Network:
    def __init__(self, il, hl, ol):
        self.inputLayer = []
        self.hiddenLayer = []
        self.outputLayer = []

        for a in range(il):
            self.inputLayer.append(Neuron())
        for b in range(hl):
            self.hiddenLayer.append(Neuron())
        for c in range(ol):
            self.outputLayer.append(Neuron())

    def printNetwork(self):
        for n in self.inputLayer:
            n.printNeuron()
        print()
        for n in self.hiddenLayer:
            n.printNeuron()
        print()
        for n in self.outputLayer:
            n.printNeuron()
        print()



if __name__ == "__main__":
    a = Neuron(1, 0, 0)
    b = Neuron(1, 0, 0)
    c = Neuron(1, 0, 0)
    d = Neuron(1, 0, 0)
    e = Neuron(1, 0, 0)
    f = Neuron(1, 0, 0)
    g = Neuron(1, 0, 0)
    h = Neuron(1, 0, 0)
    i = Neuron(1, 0, 0)
    j = Neuron(1, 0, 0)

    net = Network([a, b, c], [d, e, f], [g, h, i, j])
    net.printNetwork()
