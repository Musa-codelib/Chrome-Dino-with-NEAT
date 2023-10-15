import numpy as np
import random


#activation function for output layer

def sigmoid(x):
    return 1/(1+np.exp(-x))

def swish(x):
    temp=np.array(x)
    return temp*sigmoid(temp)
def relu(x):
    temp=np.array(x)
    return temp.clip(min=0)

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))



class layer:
    def __init__(self):
        self.input=None
        self.output=None

class FClayer:
    def __init__(self,input_size,output_size):
        self.weights=np.random.rand(input_size,output_size)-0.5
        self.bias=np.random.rand(1,output_size)-0.5

    def forward_propogation(self,input_data):
        self.input=input_data
        self.output=np.dot(self.input,self.weights)+self.bias
        self.output=self.output[0]
        return self.output[0]
        
class ActivationLayer(layer):
    def __init__(self,activation_func):
        self.activation=activation_func

    def forward_propogation(self,input_data):
        self.input=input_data
        self.output=self.activation(self.input)
        return self.output
        

#nn class which has 2 hidden layers
#hidden layers use relu activation function
#ouput layer uses sigmoid activation function
class network:
                #total inputs to our nn, final output size
    def __init__(self,input_size,output_size):
        self.layers=[]
        self.layers.append(FClayer(input_size,10))
        self.layers.append(ActivationLayer(relu))
        self.layers.append(FClayer(10,6))
        self.layers.append(ActivationLayer(relu))
        self.layers.append(FClayer(6,output_size))
        self.layers.append(ActivationLayer(sigmoid))

    def predict(self,input_data):
        result=input_data
        for lay in self.layers:
            out=lay.forward_propogation(result)
            result=out
        return result





