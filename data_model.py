import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Activation, Dense,Input, Dropout
from keras.models import Model
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint
from keras import optimizers as opt

######################################################################################################################################################################################
"""
	THIS MODEL WILL BE USED TO ALLOCATE CATEGORIES TO THE DATASET IN A UNSUPEVISED WAY. THIS MODEL USES 'KMeans'
	ALGORITHM TO ALLOCATE CATEGORIES. 
"""

def model_1(data,category) :
	dict_obj = {}

	for i in range(0,len(category)) :
		if pd.isnull(category.iloc[i]) == 'False' :
			continue
		else:
			dict_obj.__setitem__(i,category[i])

	dict_obj = dict( [(k,v) for k,v in dict_obj.items() if v!=''])
	num_clusters = list(dict_obj)[-1]

	vectorizer = TfidfVectorizer(stop_words='english',ngram_range = (2,3))
	x = vectorizer.fit_transform(data)

	model = KMeans(n_clusters=num_clusters,verbose=1,random_state=0).fit(x)
	predictions = model.predict(x)
	pred = []
	for i in predictions:
		pred.append(dict_obj.get(i))
	print(pred)
	return pred
######################################################################################################################################################################################
"""
	THIS MODEL WILL BE USED TO FURTHER CLASSIFY THE CATEGORIES TO TEST DATA BASED ON TRAINING DATA.
	IT WILL RETURN RESULT DATAFRAME.
"""

def final_model(data):
	train_size = int(len(data) * .90) ## DEFINING THE RATIO OF DATA FOR TRAINING i.e.80% DATA FOR TRAINING AND REMAINING 20% FOR TESTING

	train1 = data['id'][:train_size] ## TRAIN FEATURE 1
	train2 = data['keyword_set'][:train_size] ## TRAIN FEATURE 2

	train_label1 = data['catg_1'][:train_size] ## TRAIN LABEL 1
	train_label2 = data['catg_2'][:train_size] ## TRAIN LABEL 2

	test1 = data['id'][train_size:] ## TEST FEATURE1
	test2 = data['keyword_set'][train_size:] ## TEST FEATURE2	

	######## TOKENIZING BOTH TEST AND TRAIN DATA IN A PARTICULAR FORMAT ###############

	vocab = 500 ## VOCABULARY SIZE FOR FEATURE1
	tokenizer = Tokenizer(num_words=vocab)
	tokenizer.fit_on_texts(train2)
	x_train = tokenizer.texts_to_matrix(train2, mode='tfidf')
	x_test = tokenizer.texts_to_matrix(test2, mode='tfidf')

	"""
	vocab1 = 500 ## VOCABULARY SIZE FOR FEATURE1
	tokenizer = Tokenizer(num_words=vocab1)
	tokenizer.fit_on_texts(train1)
	x_train1 = tokenizer.texts_to_matrix(train1, mode='tfidf')
	x_test1 = tokenizer.texts_to_matrix(test1, mode='tfidf')
	"""
	######### ENCODING THE TRAIN LABELS. IF TEST LABELS ARE PRESENT THEN WE CAN ENCODE THEM ALSO. ##############
	encoder = LabelBinarizer()
	encoder.fit(train_label1)
	y_train1 = encoder.transform(train_label1)
	y_train2 = encoder.transform(train_label2)

	################################### Dense input & output layers. Creating Model #####################################################

	x_in = Input(shape=(vocab,),name = 'Keywords') ## DEFINING THE SHAPE FOR TRAIN FEATURE
	#x_in1 = Input(shape=(vocab,),name = 'Keywords') ## DEFINING THE SHAPE FOR TRAIN FEATURE

	x = Dense(164,activation = 'relu')(x_in) ## PASSING THE TRAIN FEATURE TO DENSE LAYERS FOR TRAINING
	x = Dropout(0.3)(x) 

	output1 = Dense(y_train1.shape[1],activation = 'softmax',name = 'final_out1')(x)
	output2 = Dense(y_train2.shape[1],activation = 'softmax',name = 'final_out2')(x)

	output = [output1,output2] ## output layers
	y = [y_train1,y_train2] ## output encoded data

	output_1 = []
	output_2 = []

	for i in range(0,2) :
		model = Model(inputs=x_in,outputs=output[i]) ## DEFINING THE MODEL WITH INPUT AND OUTPUT
		model.summary() ## GENERATING SUMMARY OF THE MODEL

		################################## Model Training ##################################################################

		optimizer = opt.adam(lr=0.01) ## DEFINING THE OPTIMIZER ALONG WITH LEARNING RATE
		model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy']) ## COMPILING THE DEFINED MODEL FOR MINIMIZING THE LOSS(error) AND TRACK THE ACCURACY PARAMETER
		model.fit(x_train,y[i],batch_size=120,epochs=700,shuffle=True,verbose=1) ## PASSING THE TRAINING FEATURES AND LABELS TO THE MODEL

		train_score = model.evaluate(x_train,y[i],batch_size=120, verbose=1) ## TRAINING EVALUATION ALONG WITH LOSS AND ACCURACY SCORE
		print('Train[TRAIN_LOSS,ACCURACY]:', train_score) ##### train set loss and accuracy

		############################### PREDICTION #############################################################################

		text_labels = encoder.classes_ ## ORIGINAL LABELS
		
		if i == 0:  
			for j in range(0,len(x_test)) :
				prediction = model.predict(np.array([x_test[j]]))
				predicted_label= text_labels[np.argmax(prediction[0])]
				output_1.append(predicted_label)

		else :
			for j in range(0,len(x_test)) :
				prediction = model.predict(np.array([x_test[j]]))
				predicted_label= text_labels[np.argmax(prediction[0])]
				output_2.append(predicted_label)


	store_list = []
	if len(output_1) == len(output_2):
		for i in range(0,len(output_1)):
			store_list.append((test1.iloc[i],output_1[i],'-->',output_2[i],'-->',test2.iloc[i]))

	result = pd.DataFrame(store_list)

	return result

########################################################################################################################################################################################		




	


