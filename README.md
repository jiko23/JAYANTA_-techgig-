This project is based on Keyeword/entity/key_phrases extraction from a dataset. In this project a possible approach to map Keyeword/entity/key_phrases to categories has been also demonstrated. There could be many possible solutions for mapping to categories might be present or can be build. As mapping manually is not possible for a large dataset thus, need to do automatically. The approach shown here is unsupervised method. Any suggestion for changes are always welcome.

Environment Setup for this project:
 1. Install Anaconda (python_version >= 3)
 2. Install Tensorflow-GPU and Keras. Take help of the below document:
    https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781
 3. Install googletrans package from anaconda cloud or using pip.
 4. Install textblob package from anaconda cloud or using pip.
 5. Install nltk package from anaconda cloud.
 6. Install rake_nltk.
 
 
 Running the project::
 1. 'data_preprocess.py' : It consists of data preprocessing steps,translation of data(text) to english language. For translating to       english language two different packages has been used: (a) textblob, (b) googletrans . To avoid putting much load on one translator try-except method has been used. It is recommended that internet connection must be very strong for large dataset translation. Key_phrase extraction has been also done here. Finally the preprocessed data has been converted to .csv file. (Kindly change the saving address as per yourself)
 
 2. 'data_model.py' : It consists of the AI/ML models which will be used for mapping to different categories.
 
 3. 'data_result.py' : It calls models from 'data_model.py' and generated results and converts it into a .csv file.
 
 Steps:
 Run scripts in the below order:
  1. data_preprocess.py
  2. data_result.py
