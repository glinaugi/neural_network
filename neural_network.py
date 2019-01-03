import numpy as np
import scipy.special
from keras.datasets import mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()


train_data_modify  = (train_data.reshape((60000, 28 * 28)).astype('float32') / 255.0 * 0.99) + 0.01
test_data_modify = (test_data.reshape((10000, 28 * 28)).astype('float32') / 255.0 * 0.99) + 0.01

targets = np.zeros([60000, 10]) + 0.01

i = 0
for target in targets:
    target[int(train_labels[i])] = 0.99
    i += 1



class neural_network:
    #### initialize
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr):
        self.input_node = input_nodes
        self.hidden_node = hidden_nodes
        self.output_node = output_nodes
        
        self.lr = lr
        
        
        #### weights 
        
        self.wih = np.random.normal(0.0, pow(self.input_node, -0.5), (self.input_node, self.hidden_node))
        self.who = np.random.normal(0.0, pow(self.hidden_node, -0.5), (self.hidden_node, self.output_node))
         
        
        #### activation function 
        
        self.activation = lambda x: scipy.special.expit(x)
    
    #### training    
        
    def training(self, input_list, target):
        
        iterate = len(input_list) * 2 + 5
        
        print(iterate)
        for i in range(90):
            inputs = input_list
            targets = target
        
            ##### 
        
            hidden_input = np.dot(inputs, self.wih)
            hidden_output = self.activation(hidden_input)
        
            #####
        
            final_input = np.dot(hidden_output, self.who)
            final_output = self.activation(final_input)
          

            errors = targets - final_output
            hidden_errors = np.dot(errors, self.who.T)
            
            
            dd1 = self.lr * np.transpose(np.dot((errors * final_output*(1.0 - final_output)).T, (hidden_output)))

            self.who += dd1
            
            dd2 = self.lr * np.transpose(np.dot((hidden_errors * hidden_output * (1.0 - hidden_output)).T, (inputs))) 
            self.wih += dd2
            
                        
    

    def testing(self, input_list):
        inputs = input_list
        
        hidden_input = np.dot(inputs, self.wih)
        hidden_output = self.activation(hidden_input)
    
        
        final_input = np.dot(hidden_output, self.who)

        final_output = self.activation(final_input)
        
        
        return final_output
    
    
nn = neural_network(784, 100, 10, 0.5)
    
nn.training(train_data_modify[:100], targets[:100])

a = nn.testing(train_data_modify[:100])
