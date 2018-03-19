import math


#Constants




class backpropagation:
    def __init__(self):
        self.netVal = 0
        self.entries = [[0, 0], [1, 0], [0, 1], [1, 1]]
        self.expectedValOut=[0,1,1,0]
        self.expectedValIndex=0
        self.weightsIn = [[0.1, 0.5], [-0.7, 0.3]]
        self.weightsOut = [0.2, 0.4]
        self.hiddenLayerOuts = []
        self.outputLayerOut=0
        self.error = 0
        self.learnRat = 0.25
        self.e = math.e

    #Calculate the first NET value
    def netHidden(self):
        index = 0
        for eTemp in self.entries:
            for ent in eTemp:
                for w in self.weightsIn:
                    self.netVal += w[index] * ent
                index += 1
                self.hiddenLayerOuts.append(self.outFunction(self.netVal))
            netOut()
            self.expectedValIndex+=1
    #return out function value for each hidden neuron
    def outFunction(self,value):
        return 1 / (1 + (math.pow(self.e,value)))


    #Calculates the NET value in the out layer
    def netOut(self):
        flag = 0
        netOutVal = 0
        for tempEntry in hiddenLayerOuts:
            netOutVal += self.weightsOut[flag] * tempEntry
            flag += 1
        self.outputLayerOut = self.outFunction(netOutVal)


    def calculateError(self):
        self.error=self.expectedValOut[self.expectedValIndex]- self.outputLayerOut
    def deltaFunction(self):

