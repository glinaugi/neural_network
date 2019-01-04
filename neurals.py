#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.datasets import mnist


# In[2]:


(train_input, train_label), (test_input, test_label) = mnist.load_data()


# In[3]:


train_input = (train_input.reshape(60000, 784) / 255.0 * 0.99) + 0.01

targets = np.zeros((60000, 10)) + 0.01
i = 0
for target in targets:
    target[int(train_label[i])] = 0.99
    i += 1
    


# In[4]:


class neuralnetwork:
    def __init__(self, inode , hnode, onode):
        self.input_node = inode
        self.output_node = onode
        self.hidden_node = hnode
        
        self.wih = 2 * np.random.random((self.input_node, self.hidden_node)) - 1
        self.who = 2 * np.random.random((self.hidden_node, self.output_node)) - 1
        
    def activate(self, x, deriv = False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
        
    def training(self, input_list, target):
        for i in range(1500):
            layer0 = input_list
            targets = target
        
            layer1 = self.activate(np.dot(layer0, self.wih))
            layer2 = self.activate(np.dot(layer1, self.who))
        
            errors = targets - layer2
            hidden_errors = np.dot(errors, self.who.T)
            
            self.who += np.transpose(np.dot((errors * self.activate(layer2, deriv = True)).T, layer1))
            self.wih += np.transpose(np.dot((hidden_errors * self.activate(layer1, deriv = True)).T, layer0))
                     
        #   if i % 10 == 0:
            #    print("Error: ", errors)
             #   print("outputs: ", layer2)    
             #   print()
            
                
    def testing(self, input_list):
        layer0 = input_list

        
        layer1 = self.activate(np.dot(layer0, self.wih))
        layer2 = self.activate(np.dot(layer1, self.who))
        
        return layer2
        
        


# In[10]:


nn = neuralnetwork(784, 100 , 10)
#nn.testing(testing_inputs)         


# In[11]:


for i in range(1,20):
    inputs = train_input[:i]
    target = targets[:i]

    nn.training(inputs, target)
    a = nn.testing(inputs)
    
a


# In[ ]:





# In[ ]:


inputss = np.array([
    [1,0,0,1,0,1,1,1,1,1,1,0,0,0,1,0,1,0,1],
    [1,0,0,1,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1],
    [0,1,0,1,0,1,1,0,1,1,1,0,0,0,1,0,1,0,1],
    [1,1,0,1,0,1,1,1,1,1,1,0,0,0,1,0,1,0,1],
    [1,0,0,1,0,1,1,1,1,0,1,0,0,0,1,0,1,0,1],
    [1,1,0,1,0,1,1,1,1,1,1,0,0,0,1,0,1,0,1],
    [0,0,1,1,0,1,1,1,1,1,1,0,0,0,1,0,1,0,1],
    [0,0,0,1,0,1,1,0,1,1,1,0,0,0,1,0,1,0,1]
    
    
    
])     

targetss = np.array([
    [1,0,0],
    [0,0,0],
    [0,1,0],
    [1,1,0],
    [1,0,0],
    [1,1,0],
    [0,0,1],
    [1,1,1]
    
])

testing_inputs = np.array([
    [1,1,0,1,0,1,1,1,1,1,1,0,0,0,1,0,1,0,1],
    [1,0,0,1,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1],
    [0,0,0,1,0,1,1,0,1,1,1,0,0,0,1,0,1,0,1],
    [0,1,0,1,0,1,1,0,1,1,1,0,0,0,1,0,1,0,1],
    
])

