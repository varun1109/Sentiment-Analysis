import sys
import os
import random
import math
import accuracy_of_classifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import accuracy_of_classifier as ac
import copy

#this function adds two lists of 3-dimensions
def add_lists(x, y):
	final_result = []
	for i in range(len(x)):
		layer = []
		for j in range(len(x[i])):
			neuron = []
			for k in range(len(x[i][j])):
				neuron.append(x[i][j][k] + y[i][j][k])
			layer.append(neuron)
		final_result.append(layer)

	return final_result

#all neurons have the same sigmoid function
class neuron:
	def __init__(self,a = 1.716, b = 2.0/3.0):
		self.__a = a
		self.__b = b
		self.__dervative = 0.0
		self.__output = 0.0
	
	def compute(self, input_data, func = "sigmoid"):
		if func == "sigmoid":	
			return self.__a*math.tanh(self.__b*np.array(input_data))
		if func == "linear":
			return input_data
		if func == "exp":
			return math.exp(input_data)
	
	def compute_derivative(self,input_data, func = "sigmoid"):
		if func == "sigmoid":	
			return (self.__a*self.__b)/(math.cosh(self.__b*np.array(input_data))*math.cosh(self.__b*np.array(input_data)))

	def store_derivative(self, x):
		self.__dervative = x
		return

	def get_derivative(self):	
		return self.__dervative

	def store_output(self,x):
		self.__output = x
		return

	def get_output(self):
		return self.__output

class hidden_layer:
	def __init__(self,no_of_neurons): 
		self.__neuron_list = []
		for i in range(no_of_neurons):
			self.__neuron_list.append(neuron())	

	def number_of_neurons_in_layer(self):
		return len(self.__neuron_list)

	def get_neuron(self, i):
		return self.__neuron_list[i] 

class input_layer:
	def __init__(self,no_of_neurons): 
		self.__neuron_list = []
		for i in range(no_of_neurons):
			self.__neuron_list.append(neuron())
	
	def number_of_neurons_in_layer(self):
		return len(self.__neuron_list)

	def get_neuron(self, i):
		return self.__neuron_list[i] 

class output_layer:
	def __init__(self,no_of_neurons):
		self.__neuron_list = [] 
		for i in range(no_of_neurons):
			self.__neuron_list.append(neuron())	

	def number_of_neurons_in_layer(self):
		return len(self.__neuron_list)

	def get_neuron(self, i):
		return self.__neuron_list[i] 

class neural_networks:
	def __init__(self, training_data, class_labels, no_of_hidden_layers = 1, initial_weight = 1e-10, number_of_nodes_in_hidden_layer_input = 3): 
		#instance variables
		self.__training_data = training_data 
		self.__class_labels = class_labels
		no_of_input_neurons = training_data.shape[1]
		#print training_data
		self.__input_layer = input_layer(no_of_input_neurons + 1)
		self.__no_of_hidden_layers = no_of_hidden_layers
		self.__output_layer = output_layer(len(set(self.__class_labels)))
		
		#setting target vectors
		self.__class_index = {} #mapping class label to index
		self.__class_name = {} #mapping index to class label
		index = 0
		for i in set(self.__class_labels):
			self.__class_index[i] = index
			self.__class_name[index] = i
			index += 1		
		
		#this stores each hidden layer 
		self.__hidden_layers = []
		
		for i in range(no_of_hidden_layers):
			no_of_input_neurons /= 10
			if(no_of_input_neurons == 0): #too many hidden layers not required, based on heuristics.
				no_of_input_neurons = 3
			self.__hidden_layers.append(hidden_layer(number_of_nodes_in_hidden_layer_input + 1))
		
		
		#to maintain count of the hidden layer for creating the network	
		hidden_index = 0
	
		#this is a list containing weight vectors from each neuron of lower layer to upper layer  
		# structure of self.__neuron_layer -->[i,j,k] (layer,neuron,weight to neuron k in upper layer)
		self.__neuron_layer = []

		#initializing all weights to some value

		#for input layer
		weights = {}
		for i in range(self.__input_layer.number_of_neurons_in_layer()):
			for j in range(self.__hidden_layers[hidden_index].number_of_neurons_in_layer() - 1): #as last one is bias node
				if i not in weights:
					weights[i] = []
				weights[i].append(random.uniform(-initial_weight,initial_weight))
				#weights[i].append(initial_weight)

		self.__neuron_layer.append(weights)

		#for hidden layer
		while (hidden_index + 1 < self.__no_of_hidden_layers):
			weights = {}
			for i in range(self.__hidden_layers[hidden_index].number_of_neurons_in_layer()):
				for j in range(self.__hidden_layers[hidden_index + 1].number_of_neurons_in_layer() - 1): #as last one is bias node
					if i not in weights:
						weights[i] = []
					weights[i].append(random.uniform(-initial_weight,initial_weight))
					#weights[i].append(initial_weight)

			self.__neuron_layer.append(weights)
			hidden_index += 1

		#for output layer
		weights = {}
		for i in range(self.__hidden_layers[hidden_index].number_of_neurons_in_layer()): #the last one is a pseudo neutron for the bias weight
			for j in range(self.__output_layer.number_of_neurons_in_layer()):
				if i not in weights:
					weights[i] = []
				weights[i].append(random.uniform(-initial_weight,initial_weight))
				#weights[i].append(initial_weight)
		
		self.__neuron_layer.append(weights)
				

	def feed_forward(self, input_data):
		#list of all nets. net[0] is the input to 0th node in upper layer.
		net = []
		#input to hidden layer 1
		for i in range(self.__hidden_layers[0].number_of_neurons_in_layer() - 1):
			netsum = 0
			#print self.__input_layer.number_of_neurons_in_layer()
			
			for j in range(self.__input_layer.number_of_neurons_in_layer() - 1): 
				neuron_output = self.__input_layer.get_neuron(j).compute(input_data[0,j],"linear")
				netsum += self.__neuron_layer[0][j][i]*neuron_output
				self.__input_layer.get_neuron(j).store_output(neuron_output)
				
			netsum += self.__neuron_layer[0][j][i] #adding the bias		
			net.append(netsum)

	
		#for hidden layers
		for k in range(1,len(self.__neuron_layer) - 1):
			input_for_nodes = net[:]
			net = []
			for i in range(self.__hidden_layers[k].number_of_neurons_in_layer() - 1):
				netsum = 0
				for j in range(self.__hidden_layers[k - 1].number_of_neurons_in_layer() - 1):
					neuron_output = self.__hidden_layers[k - 1].get_neuron(j).compute(input_for_nodes[j],"sigmoid")	
					netsum += self.__neuron_layer[k][j][i]*neuron_output
					self.__hidden_layers[k - 1].get_neuron(j).store_output(neuron_output)
					self.__hidden_layers[k - 1].get_neuron(j).store_derivative(self.__hidden_layers[k - 1].get_neuron(j).compute_derivative(input_for_nodes[j],"sigmoid"))	#storing the derivative of net at each neuron
				netsum += self.__neuron_layer[k][j][i] #adding the bias			
				net.append(netsum)	
			
	
		input_for_nodes = net[:]
		net = []
		#for hidden layer to output layer	
		for i in range(self.__output_layer.number_of_neurons_in_layer()):
			for j in range(self.__hidden_layers[len(self.__hidden_layers) - 1].number_of_neurons_in_layer() - 1): 
				neuron_output = self.__hidden_layers[len(self.__hidden_layers) - 1].get_neuron(j).compute(input_for_nodes[j],"sigmoid")
			#	print i, j, self.__output_layer.number_of_neurons_in_layer()
				netsum += self.__neuron_layer[len(self.__hidden_layers)][j][i]*neuron_output
				self.__hidden_layers[len(self.__hidden_layers) - 1].get_neuron(j).store_output(neuron_output)
				self.__hidden_layers[len(self.__hidden_layers) - 1].get_neuron(j).store_derivative(self.__hidden_layers[len(self.__hidden_layers) - 1].get_neuron(j).compute_derivative(input_for_nodes[j],"sigmoid"))	#storing the derivative of net at each neuron

			netsum += self.__neuron_layer[len(self.__hidden_layers)][j][i] #adding the bias		
			net.append(netsum)
		
		output = []
		#for final outputs
		for i in range(self.__output_layer.number_of_neurons_in_layer()):
			output.append(self.__output_layer.get_neuron(i).compute(net[i],"sigmoid"))
			self.__output_layer.get_neuron(i).store_derivative(self.__output_layer.get_neuron(i).compute_derivative(net[i],"sigmoid"))
		
		return output		
	
	def calc_error(self,output, index):
		error = 0.0
		#print index, self.__class_labels[index], self.__class_index[self.__class_labels[index]] 
		for j in range(len(output)):
			target = -1 #the target value for each output node, is 1 if that output node represents the actual class!
			if self.__class_index[self.__class_labels[index]] == j:
				target = 1
			error += ((output[j] - target)**2)*0.5
		return error

	def calc_error_total(self,output, class_labels):
		error = 0.0
		for i in range(len(class_labels)):
			index = 0
			for j in output[i]:
				target = -1 #the target value for each output node, is 1 if that output node represents the actual class!
				if self.__class_index[class_labels[i]] == index:
					target = 1
				error += ((j - target)**2)*0.5
				index += 1
		return error
	

	def back_propagation(self, output, index, learning_rate = 0.1):
		sensitivities = []
		#weight updates per layer
		layer_no = []

		#this will contain the individual weight updates will be added to layer_no
		temp_layer = []
		
		#hidden to output weights error propagation
		for i in range(self.__output_layer.number_of_neurons_in_layer()):
			temp = []
			#setting up the target vector
			target = -1.0
			if(self.__class_index[self.__class_labels[index]] == i):
				target = 1.0
			
			sensitivities.append(self.__output_layer.get_neuron(i).get_derivative()*(target - output[i]))
			for j in range(self.__hidden_layers[len(self.__hidden_layers) - 1].number_of_neurons_in_layer() - 1):
				temp.append(learning_rate*self.__output_layer.get_neuron(i).get_derivative()*(target - output[i])*(self.__hidden_layers[len(self.__hidden_layers) - 1].get_neuron(j).get_output()))

			#correcting the bias
			temp.append(learning_rate*self.__output_layer.get_neuron(i).get_derivative()*(target - output[i]))

			temp_layer.append(temp)	
		layer_no.append(temp_layer)	
		
		temp_layer = []

		#hidden to hidden/input weights error propagation
		for k in list(reversed(range(len(self.__hidden_layers)))):
			#this will contain the individual weight updates will be added to layer_no
			temp_layer = []
			old_sensitivities = sensitivities[:]
			sensitivities = []
			
			
			for i in range(self.__hidden_layers[k].number_of_neurons_in_layer() - 1):
				temp = []
				sensitivities_from_previous_layer = 0
				if(k < len(self.__hidden_layers) - 1):
					for l in range(self.__hidden_layers[k + 1].number_of_neurons_in_layer() - 1):
							sensitivities_from_previous_layer += old_sensitivities[l]*self.__neuron_layer[k + 1][i][l]

				else:
					for l in range(self.__output_layer.number_of_neurons_in_layer()):
							sensitivities_from_previous_layer += old_sensitivities[l]*self.__neuron_layer[k + 1][i][l]

				sensitivities.append(sensitivities_from_previous_layer*self.__hidden_layers[k].get_neuron(i).get_derivative())
				
				if(k > 0):
					for j in range(self.__hidden_layers[k - 1].number_of_neurons_in_layer() - 1):	
						correction = learning_rate*self.__hidden_layers[k - 1].get_neuron(j).get_output()*self.__hidden_layers[k].get_neuron(i).get_derivative()*sensitivities_from_previous_layer
					
						temp.append(correction)	
				
				else:
					for j in range(self.__input_layer.number_of_neurons_in_layer() - 1):	
						correction = learning_rate*self.__input_layer.get_neuron(j).get_output()*self.__hidden_layers[k].get_neuron(i).get_derivative()*sensitivities_from_previous_layer
					
						temp.append(correction)	
				#correcting the bias
				temp.append(learning_rate*self.__hidden_layers[k].get_neuron(i).get_derivative()*sensitivities_from_previous_layer)
				
				temp_layer.append(temp)	

			layer_no.append(temp_layer)	
		return layer_no

	def learn(self, validation_set, validation_set_class_labels, learning_rate = 0.001): 
		gradient = 10
		number_of_corrected_samples = 0
		super_old_learning_rate = learning_rate
		constant_learning_rate_counter = 0
		epoch = 0
		minimum_error_till_now = 10000 # for validation-set error
		first_time = True #for saving the weight_parameters
		best_weights = []
		#print validation_set, validation_set_class_labels
	 	while(1):
			if(number_of_corrected_samples == len(self.__training_data)):
				break
			avg_error = 0
			max_gradient = -100000 #initialize at a random starting value
			if (abs(super_old_learning_rate - learning_rate) < 1e-8):
				constant_learning_rate_counter += 1
			else:
				super_old_learning_rate = learning_rate
				constant_learning_rate_counter = 0


			if (constant_learning_rate_counter >= 5):
				learning_rate *= 2
				constant_learning_rate_counter = 0
			#feedforward
			for i in range(len(self.__training_data)):	
				output = self.feed_forward(self.__training_data[i])
	
				#calculating the total error
				error = self.calc_error(output, i)
				old_error = error

				#this variable is to tell the gradient descent to go ahead even if it increases error to prevent being stuck
				go_ahead = False
				while(1):
					#backpropagation
					weight_corrections = self.back_propagation(output, i, learning_rate)	
				
					#correcting the weights
					number_of_layers = len(self.__neuron_layer)
					ii = number_of_layers - 1
					j = 0
			
			
					old_weights = copy.deepcopy(self.__neuron_layer[:])
			
					while(ii >= 0):
						#print i, "haha"
						for n in range(len(self.__neuron_layer[ii])):
							#print n
							for u in range(len(self.__neuron_layer[ii][n])):
								#print u
								self.__neuron_layer[ii][n][u] = self.__neuron_layer[ii][n][u] + weight_corrections[j][u][n] 				
						
						ii -= 1
						j += 1
			

					#checking error
					output_new = self.feed_forward(self.__training_data[i])
					error = self.calc_error(output_new, i)

					#print old_error, error, learning_rate
				
					#calculating the gradient
					m = max(weight_corrections)
					k = weight_corrections.index(m)
					gradient = float(max(weight_corrections[k][weight_corrections[k].index(max(weight_corrections[k]))]))/learning_rate

		
					if(old_error < error and go_ahead == False):
						learning_rate /= 2.0
						
						if(gradient*learning_rate < 1e-9 and gradient > 1e-7):
							learning_rate = 1e-1
							go_ahead = True
		
						else:
							self.__neuron_layer = copy.deepcopy(old_weights[:])
							test_output = self.feed_forward(self.__training_data[i])
							error = self.calc_error(test_output, i)
					
					else:
						#print i, error, learning_rate,
						avg_error += error
						max_gradient = max(max_gradient, gradient)
						break

				if (gradient < 1e-7):
					number_of_corrected_samples += 1
				else:
					number_of_corrected_samples = 0
				
			print epoch, avg_error/len(self.__training_data)# max_gradient, 
			epoch += 1





			#measure error over validation set
			final_outputs = []
			for iii in validation_set:
				final_outputs.append(self.feed_forward(iii))
				
			validation_error = self.calc_error_total(final_outputs, validation_set_class_labels)
			print "(Validation Error:", float(validation_error)/len(validation_set),
							

			#measure accuracy over validation_set
			classified_labels = []	
			actual_labels = []
			for iii in range(len(validation_set_class_labels)):
				ind = final_outputs[iii].index(max(final_outputs[iii]))	
				classified_labels.append(self.__class_name[ind])
				actual_labels.append(validation_set_class_labels[iii])
			print "Test acc: ", accuracy_of_classifier.accuracy(actual_labels,classified_labels), ")",


				
			if(first_time == True):
				#this contains the weights which gives best results for the validation set!
				best_weights = copy.deepcopy(self.__neuron_layer)
				if(epoch > 50): #initial highly random can't rely on it
					first_time = False
				minimum_error_till_now = float(validation_error)/len(validation_set)

			else:
				if((float(validation_error)/len(validation_set)) < minimum_error_till_now):
					best_weights = copy.deepcopy(self.__neuron_layer)
					minimum_error_till_now = float(validation_error)/len(validation_set)

			#measure accuracy over training data
			final_outputs = []
			for iii in self.__training_data:	
				final_outputs.append(self.feed_forward(iii))

			classified_labels = []	
			actual_labels = []
			for iii in range(len(self.__class_labels)):
				ind = final_outputs[iii].index(max(final_outputs[iii]))	
				classified_labels.append(self.__class_name[ind])
				actual_labels.append(self.__class_labels[iii])
			print accuracy_of_classifier.accuracy(actual_labels,classified_labels)
			#print epoch
			if(avg_error/len(self.__training_data) <= 0.5 or epoch > 500):
				break
		

######################################################################################
		#print "hola"		
		#measure accuracy over training data
		"""final_outputs = []
		for i in self.__training_data:	
			final_outputs.append(self.feed_forward(i))

		classified_labels = []	
		actual_labels = []
		for i in range(len(self.__class_labels)):
			ind = final_outputs[i].index(max(final_outputs[i]))	
			classified_labels.append(self.__class_name[ind])
			actual_labels.append(self.__class_labels[i])
		print accuracy_of_classifier.accuracy(actual_labels,classified_labels)"""
		
		self.__neuron_layer = copy.deepcopy(best_weights)	

		"""#measure accuracy over training data
		final_outputs = []
		for i in self.__training_data:	
			final_outputs.append(self.feed_forward(i))

		classified_labels = []	
		actual_labels = []
		for i in range(len(self.__class_labels)):
			ind = final_outputs[i].index(max(final_outputs[i]))	
			classified_labels.append(self.__class_name[ind])
			actual_labels.append(self.__class_labels[i])
		print accuracy_of_classifier.accuracy(actual_labels,classified_labels)"""
		
		#measure accuracy over testing data
		final_outputs = []
		for iii in validation_set:	
			final_outputs.append(self.feed_forward(iii))
		validation_error = self.calc_error_total(final_outputs, validation_set_class_labels)
		print "test error: ", float(validation_error)/len(validation_set),

		classified_labels = []	
		actual_labels = []
		for iii in range(len(validation_set_class_labels)):
			ind = final_outputs[iii].index(max(final_outputs[iii]))	
			classified_labels.append(self.__class_name[ind])
			actual_labels.append(validation_set_class_labels[iii])
		print "test acc: ", accuracy_of_classifier.accuracy(actual_labels,classified_labels),
		

	def classify(self, x): 
		#feedforward
		output = self.feed_forward(x)
		ind = output.index(max(output))
		#print output
		return self.__class_name[ind]

