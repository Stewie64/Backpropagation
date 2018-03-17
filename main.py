import math


#Constants
entries = [[0, 0], [1, 0], [0, 1], [1, 1]]
weightsIn = [[0.1, 0.5], [-0.7, 0.3]]
weightsOut = [0.2, 0.4]
learnRat = 0.25
e = math.e
outH_Array = []


class backpropagation:
    def __init__(self):
        self.netVal = 0

    #Calculate the first NET value
    def netHidden(self):
        index = 0
        for eTemp in entries:
            for e in eTemp:
                for w in weightsIn:
                    self.netVal += w[index] * e
                index += 1
                outH_Array.append(self.outFunction(self.netVal))

    #return out function value for each hidden neuron
    def outFunction(self,value):
        return 1 / (1 + (math.pow(e,value)))


    #Calculates the NET value in the hidden layer
    def netOut(self):
        flag = 0
        netOutVal = 0
        for tempEntry in outH_Array:
            netOutVal += weightsOut[flag] * tempEntry
            flag += 1
            return self.outFunction(netOutVal)