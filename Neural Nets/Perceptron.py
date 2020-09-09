'''

---------------------------------------------------------

Domingo Cajina 08/09/2020

---------------------------------------------------------


Nuestra clase Perceptron tendra 3 metodos:

1.- Inicializar el perceptron
	Pesos iniciales aleatorios

2.- Calculo de salida del perceptron

3.- Entrenamiento


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Perceptron():

	def __init__(self, n):
		'''

		Funcion constructor, inicializa los valores iniciales
		de los pesos y guardamos la dimension del vector de entrada.

		'''
		self.weights = np.random.randn(n)
		self.n = n

	def sigmoid(self, x):
		'''

		Funcion de activacion de nuestro perceptron

		'''
		return 1/(1 + np.exp(-x))

	def _Propagation_(self, inputs):
		'''

		Calculamos el producto punto,
		entre los pesos y nuestras entradas.

		Retornamos la funcion sigmoide evaluada
		en el resultado del producto punto.

		'''
		self.inputs = inputs
		pp = self.weights.dot(inputs)
		self.out = self.sigmoid(pp)

	def _Fitness_(self, learning_rate, target):
		'''
		Funcion de entrenamiento para el perceptron,
		ajustamos los pesos obteniendo el error.

		Entrenamos un determinado numero de epocas.

		'''
		for i in range(0, self.n):
			self.weights[i] = self.weights[i] + learning_rate * ( target - self.out) * self.inputs[i]


'''
TESTING

'''

def main():

	print('TESTING')
	#Instanciamos nuestro perceptron
	p = Perceptron(3)

	#Vemos los pesos iniciales
	print(p.weights)

	#Damos la entrada a nuestra neurona
	p._Propagation_([1,2,1])

	#Observamos la salida de la misma
	print(p.out)

	#Entrenamos una vez para confirmar que funciona
	p._Fitness_(0.8, 10)

	#Comparamos pesos anteriores con el nuevo
	print(p.weights)



	'''
	Problema:
	Cree un perceptron que aprenda la funcion logica AND

	Tabla de verdad:

	0 | 0 | 1
	0 | 1 | 0
	1 | 0 | 1
	1 | 1 | 1

	'''

	print('\n')
	print('SOLUCION PERCEPTRON COMPUERTA LOGICA AND')

	perceptron_and = Perceptron(3)

	#Creamos set de entrenamiento

	training_set = np.array([

		[0, 0, 1],
		[0, 1, 0],
		[1, 0, 0],
		[1, 1, 1]

		])

	#Guardamos los pesos
	hist_weights = [perceptron_and.weights]

	#Entrenamos nuestra neurona un numero determinado de epocas

	epochs = 100

	for k in range(epochs):
		#Tenemos 3 entradas en nuestra neurona
		for i in range(0, 4):

			perceptron_and._Propagation_(
				training_set[i, 0 : 3]
				)

			perceptron_and._Fitness_(0.8, training_set[i, 2])

			hist_weights = np.concatenate((hist_weights, [perceptron_and.weights]), axis = 0)

	plt.title('Evolucion pesos sinapticos PERCEPTRON para solucion compuerta AND')
	plt.plot(hist_weights[:,0], 'k')
	plt.plot(hist_weights[:,1], 'r')
	plt.plot(hist_weights[:,2], 'b')
	plt.xlabel('Epocas de entrenamiento')
	plt.ylabel('Valor de convergencia')
	plt.show()




if __name__ == '__main__':
	main()
	
