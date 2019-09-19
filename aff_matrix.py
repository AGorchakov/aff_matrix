#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from mpi4py import MPI 
#import mpi4py 

import time
import sys

#https://mpi4py.readthedocs.io/en/stable/tutorial.html

def shift(arr):
	arr2 = []
	for k in range(3, 12):
		arr2.append(arr[k])
	for k in range(0, 3):
		arr2.append(arr[k])
	for k in range(4, 12, 3):
		arr2[k] = np.fmod(arr2[k]-arr2[1], arr2[k+1])
	arr2[1] = 0        
	return arr2

def displ(arr, i):
	arr2 = arr
	for k in range(4, 12, 3):
		arr2[k] = np.fmod(arr[k]+i*arr[2], arr[k+1])
	return arr2

def cdist(arr1,arr2):
	sum = 0;
	for i in range(0, len(arr1)):
		sum += (arr1[i] - arr2[i])**2
	return np.sqrt(sum)

def minusE2(arr1,arr2):
	sum = 0;
	for i in range(0, len(arr1)):
		sum += (arr1[i] - arr2[i])**2
	return -sum

	
def minusR2(P1, P2, MDispl):
	minR = cdist(P1, P2)
	for m in range(MDispl):
		P3 = displ(P2, m)
		R = cdist(P1, P3)
		if R < minR:
			minR = R
		for i in range(3):
			P3 = shift(P3)
			R = cdist(P1, P3)
			if R < minR:
				minR = R
	return -minR*minR

def main():  
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()	
	comm_size = comm.Get_size()
	print(rank, comm_size)
	#Open the file back and read the contents
	train = pd.read_csv("rosen_1000_conf.txt", header=None, sep="\,\[|,|]", engine='python').drop([0,13], axis=1)
	N = train.shape[0]	
	affmatrix = np.zeros((N, N))
	LF = 0.0 #Frobenius Norm
	if rank == 0:
		start = time.time()
		for i in range(N):
			for j in range(i+1, N):
				affmatrix[i, j] = minusE2(train.iloc[i,:].tolist(), train.iloc[j,:].tolist())
				affmatrix[j, i] = affmatrix[i, j]
				LF = LF + 2.0*affmatrix[j, i]
		LF = np.sqrt(-LF)
		end = time.time()
		print("Euclidean metrics, LF=", LF, ", ", str(np.round(end - start, 3)), " seconds to count")
	else:
		start = time.time()
		for i in range(train.shape[0]):
			for j in range(i+1, train.shape[0]):
				affmatrix[i, j] = minusR2(train.iloc[i,:].tolist(), train.iloc[j,:].tolist(), rank)
				affmatrix[j, i] = affmatrix[i, j]	
				LF = LF + 2.0*affmatrix[j, i]
		LF = np.sqrt(-LF)
		end = time.time()
		print("Tor metrics (", rank, ") LF=", LF, ", ", str(np.round(end - start, 3)), " seconds to count")
if __name__== "__main__":
	main()

