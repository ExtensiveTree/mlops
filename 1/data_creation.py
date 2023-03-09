import csv
import requests
import os
import math

DS_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
TRAIN_DIR = 'train'
TEST_DIR = 'test'

def create_data(data, train_file='train.csv', test_file='test.csv', ratio=0.8,):
	if 0 <= ratio >= 1: return 'Error of ratio'

	createDirs([TEST_DIR, TRAIN_DIR])

	train_start = 1
	test_start = train_end = math.floor( (len(data) - 1) * ratio )
	test_end = (len(data) - 1)

	write_to_csv_file(TRAIN_DIR + '/' + train_file, data, train_start, train_end)
	write_to_csv_file(TEST_DIR + '/' + test_file, data, test_start, test_end)


def write_to_csv_file(file_name, data, start, end):
	with open(file_name, 'w') as f:
		f.write(data[0] + ',\n')
		for i in range(start, end):
			f.write(data[i] + ',\n')


def createDirs(dirNames):
	for dirName in dirNames:
		if not os.path.exists(dirName):
			os.makedirs(dirName)

if __name__ == "__main__":
	response = requests.get(DS_URL)  
	decoded_content = response.content.decode('utf-8').split('\n')

	create_data(decoded_content)
	print('All success')
