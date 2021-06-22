"""
Deep Residual Regression Networks
"""
import keras
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import Average,Flatten,AveragePooling1D, Input, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
K.set_learning_phase(1)
from tensorflow.keras import layers,models
from tensorflow.keras import callbacks
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import scipy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat, savemat
import pickle
import os


def abs_backend(inputs):
    return K.abs(inputs)

def expand_dim_backend(inputs):
    return K.expand_dims(K.expand_dims(inputs ,1) ,1)

def identity_block(input_tensor,units):
	"""The identity block:
	# Arguments
		input_tensor: input tensor
		units:output shape
	# Returns
		Output tensor for the block.
	"""
	x = layers.Dense(units)(input_tensor)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)

	x = layers.add([x, input_tensor])
	x = layers.Activation('relu')(x)

	return x

def dens_block(input_tensor,units):
	"""A block with dense layer at shortcut.
	# Arguments
		input_tensor: input tensor
		unit: output tensor shape
	# Returns
		Output tensor for the block.
	"""
	x = layers.Dense(units)(input_tensor)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)

	# Calculate global means
	abs_mean = K.abs(x) 
	mm=K.mean(abs_mean)

	# Calculate scaling coefficients
	scales = layers.Dense(units)(abs_mean)
	scales = layers.BatchNormalization()(scales)
	scales = layers.Activation('relu')(scales)
	scales = layers.Dense(units,activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)

	# Calculate soft-threshold for denoising
	thres = layers.multiply([abs_mean, scales])
	sub = layers.subtract([abs_mean, thres])
	zeros = layers.subtract([sub, sub])
	n_sub = layers.maximum([sub, zeros])

	# Short connection in residual unit and combine the path
	residual = layers.multiply([K.sign(x), n_sub])
	shortcut = layers.Dense(units)(input_tensor)
	shortcut = layers.BatchNormalization()(shortcut)

	x = layers.add([residual, shortcut])

	x = layers.Activation('relu')(x)

	return x


def ResNet50Regression():
	"""ResNet50 architecture.
	# Arguments        
		input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
			to use as input for the model.        
	# Returns
		A Keras model instance.
	"""
	Res_input = layers.Input(shape=(10,)) #set number of input neurons to be the number of input signal

	width = 128

	x = dens_block(Res_input,width)
	x = identity_block(x,width)
	x = identity_block(x,width)

	x = dens_block(x,width)
	x = identity_block(x,width)
	x = identity_block(x,width)

	x = dens_block(x,width)
	x = identity_block(x,width)
	x = identity_block(x,width)	

	x = dens_block(x,width)
	x = identity_block(x,width)
	x = identity_block(x,width)

	x = layers.BatchNormalization()(x)
	x = layers.Dense(8, activation='linear')(x) #set number of output to number of parameters to be predicted (from your model fitting)
	model = models.Model(inputs=Res_input, outputs=x)

	return model

################################# Prepare data ####################################


print('Loading training set...')
x_train = scipy.io.loadmat('training_data/database_train_DL_randomdire_GPD_fitT2s_SNR.mat')
TrainSig = x_train['database_train_noisy']
TrainParam = x_train['params_train_noisy']
print('Setting up the model...')

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
scaler.fit(TrainParam)
TrainParam=scaler.transform(TrainParam)

############################## Build Model ################################
model = ResNet50Regression()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.summary()

#compute running time
starttime = datetime.datetime.now()

print('Training the model...')
#set epoch to 100 or 1000
history = model.fit(TrainSig,TrainParam, epochs=10, batch_size=100, verbose=2, callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=20,verbose=2, mode='auto')], validation_split=0.2)

endtime = datetime.datetime.now()

############################## Save Model #################################
print('Saving the trained DL model...')
model.save('trained_resreg.h5')
filename = 'scaler_resreg.sav'
pickle.dump(scaler, open(filename, 'wb'))
print('DONE')

############################# Model Prediction #################################

import tensorflow as tf

# Load the trained model and scaler
model=tf.keras.models.load_model('trained_resreg.h5') 
scaler = pickle.load(open('scaler_resreg.sav', 'rb'))

# Predict the parameters from DW-MRI input
# Input is the pre-processed DW-MRI 
x_test = scipy.io.loadmat('training_data/patient_005_ROI_DL.mat')
TestSig = x_test['Signal']
TestPredict= model.predict(TestSig)
TestPredict = scaler.inverse_transform(TestPredict)

# Save prediction
data = {}
data['DLprediction'] = TestPredict
scipy.io.savemat('patient_005_pred.mat',data)
print('saved pred')