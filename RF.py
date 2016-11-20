import json
import os
import pickle
import sklearn
import random
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from random import shuffle

random.seed(123)
class RF(object):
	#data = []
	def __init__(self):
		self.size = 0
		self.data = []
		self.trainX = numpy.array([])
		self.trainY = []
		self.testX = numpy.array([])
		self.testY = []
		self.macs = []

	def get_data(self, fname):
		X = []
		item = {}
		if fname.endswith(".txt"):
			with open(fname, 'r') as f_in:
				for line in f_in:
					X.append(json.loads(line))
			for i in X:
				item["wifi-fingerprint"] = i["wifi-fingerprint"]
				item["location"] = i["location"]
				self.data.append(item)
				for j in i["wifi-fingerprint"]:
					self.macs.append(j["mac"])
		self.size = len(self.data)
		self.macs = set(self.macs)
		self.macs = list(self.macs)
		#data = self.data
		print(self.size)
		print(len(self.macs))
		return self.data
	#def get_train_data(unf_data):



	def splitDataset(self, dataset, splitRatio):
		trainSize = int(len(dataset)*splitRatio)
		self.trainX.shape=(0, len(self.macs))
		self.testX.shape=(0, len(self.macs))
		index = 0
		xs = [i for i in range(len(dataset))]
		shuffle(xs)
		while index < len(xs):
			if index < trainSize:
				item = numpy.array([self.makeMatrix(dataset, index)])
				#print(item.shape)
				#print(self.trainX.shape)
				self.trainX = numpy.concatenate((self.trainX, item),axis=0)
				self.trainY.append(dataset[xs[index]]["location"])
			else:
				item = numpy.array([self.makeMatrix(dataset, index)])
				self.testX = numpy.concatenate((self.testX, item),axis=0)
				self.testY.append(dataset[xs[index]]["location"])
			index += 1
		print(self.trainX.shape)
		print(len(self.trainY))
		print(self.testX.shape)
		print(len(self.testY))

	def makeMatrix(self, dataset, index):
		item = [] 
		dataT = dataset[index]
		dataTest = dataT["wifi-fingerprint"]
		value = -1
		for i in range(len(self.macs)):
			for j in range(len(dataTest)):
				if self.macs[i] == dataTest[j]["mac"]:
					value = dataTest[j]["rssi"] 
					break
				else:
					value = 0
			item.append(value)
		#print(item)
		#print(len(item))
		return item

	def randomF():
		clf = RandomForestClassifier(n_estimators=10, max_depth=None, 
								min_samples_split=2, random_state=0)
		
randomF = RF()
data = randomF.get_data("data/learning.txt")
#print(randomF.data)
#randomF.makeMatrix(data, 0)
randomF.splitDataset(data, 0.75)
