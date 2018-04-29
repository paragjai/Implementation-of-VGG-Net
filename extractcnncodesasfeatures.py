from keras import backend as K

class ExtractCNNCodeAsFeatures:
	def __init__(self, model, layer_index):
		# layer_index according to trained keras model. This layer's output will be used as the CNN code.
		self.model = model
		self.layer_index = layer_index
		self.derived_info()
		
	def derived_info(self):
	
		# self.model.summary()
		# print("model.layers")
		# print(self.model.layers)
		# print("len(model.layers)")
		# print(len(self.model.layers))
		# print("model.layers[0].output")
		# print(self.model.layers[0].output)
		# print("model.layers[-1].output")
		# print(self.model.layers[-1].output)
		# print("model.input")
		# print(self.model.input)

		inp = self.model.input
		
		#outputs = [layer.output for layer in self.model.layers] # Uncomment if you want to see the output of each of the layers.
		
		# keras.backend.function(inputs, outputs, updates=None)
		# Instantiates a keras function.
		# inputs : list of placeholder tensors.
		# outputs : list of output tensors.
		# returns a numpy array

		# print("K.learning_phase()")
		# print(K.learning_phase())
		# print(" [inp]+ [K.learning_phase()]")
		# print([inp] + [K.learning_phase()])

		# K.learning_phase() is required as many layers such as Dropout and BatchNormalization work differently during testing and training phase
		#self.functors = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs] # Uncomment if you want to see the output of each of the layers.

		desired_layer = self.model.layers[self.layer_index].output
		self.functor = K.function([inp] + [K.learning_phase()], [desired_layer])
		
	# image that we receive would have been expanded by one dimension using expanddimpreprocessor.py
	def preprocess(self, image):
		#print("calling preprocess of extractcnncodeasfeatures.py")
		desired_layer_out = self.functor([image, 0.])
		extracted_cnn_code = desired_layer_out[0][0]
		#print("extracted_cnn_code")
		#print(extracted_cnn_code)
		#print("extracted_cnn_code.shape")
		#print(extracted_cnn_code.shape)
		# If we were doing for ALL the layers then it would have been helpful. Remove if 0 and indent appropriately if you want to see the output of each of the layers. Also comment desired_layer_out and extracted_cnn_code.
		if 0:
			layer_outs = [func([image, 0.]) for func in self.functors]
			#print(layer_outs)
			
			print("len (layer_outs):", len(layer_outs))
			layer_outputs = []
			for layer_out in layer_outs:
				print(layer_out[0][0].shape, end = "\n.............\n")
				layer_outputs.append(layer_out[0][0])
				
			
			print("layer_outputs[0]")
			print(layer_outputs[0])
			print("layer_outputs[0].shape")
			print(layer_outputs[0].shape)

			feature_vectors = []

			for i in range(len(layer_outs)):
				if len(layer_outputs[i].shape)==3:
					if 0:
						x_max = layer_outputs[i].shape[0]
						y_max = layer_outputs[i].shape[1]
						n     = layer_outputs[i].shape[2]
						
						L = []
						for j in range(n):
							L.append(np.zeros((x_max, y_max)))
						
						for k in range(n):
							for x in range(x_max):
								for y in range(y_max):
									L[k][x][y] = layer_outputs[i][x][y][k]
								
						count = 0
						for m,img in enumerate(L):
							#plt.figure()
							#plt.imshow(img, interpolation = 'nearest')
							cv2.imshow("img_layer_"+str(m)+"_"+str(count), img)
							cv2.waitKey(0)
							cv2.destroyAllWindows()
							print("img_layer_"+str(m)+"_"+str(count)+".jpg shape", img.shape)
							cv2.imwrite("img_layer_"+str(m)+"_"+str(count)+".jpg", img)
							count = count + 1
							
				elif len(layer_outputs[i].shape)==1:
					print("len of shape is not equal to 3")
					print("layer with index : ", i)
					print("layer_outputs["+str(i)+"].shape")
					print(layer_outputs[i].shape)
					print(layer_outputs[i], end = "\n.............\n")
					feature_vectors.append(layer_outputs[i])
				else:
					print("len of shape is not equal to 3 or 1")
					print("layer_outputs["+str(i)+"].shape: ", layer_outputs[i].shape)

			
		#print("extracted_cnn_code")
		#print(extracted_cnn_code)
		return extracted_cnn_code