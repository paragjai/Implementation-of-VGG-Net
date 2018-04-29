import pickle

fileObject = open("store_feature_vectors.txt", "rb") #"b" is required else it results in an error. # TypeError: a bytes-like object is required, not 'str'
fileContentFromPickledFile = pickle.load(fileObject)
print("----File content from Pickled File----")
print(fileContentFromPickledFile) 
print("---- len(fileContentFromPickledFile) ----")
print(len(fileContentFromPickledFile))
print("---- type(fileContentFromPickledFile[0]) ----")
print(type(fileContentFromPickledFile[0]))
print("---- type(fileContentFromPickledFile[1]) ----")
print(type(fileContentFromPickledFile[1])) #restores it as a numpy array

fileObject.close()

