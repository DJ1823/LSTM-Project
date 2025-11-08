# LSTM-Project
This project presents the development of a next word prediction model using a Long Short- Term Memory (LSTM) neural network. The model is trained on a custom text dataset to predict the next word in a given sequence based on previous context. The dataset was preprocessed through tokenization, sequence padding, and one-hot encoding to prepare it for training. The system was implemented in Google Colab using TensorFlow and Keras, and trained for 100 epochs to optimize performance. The results demonstrate that the LSTM model effectively learns word patterns and context, achieving reliable accuracy in predicting the next word. This work shows the potential of deep learning in enhancing text generation and natural language processing applications such as chatbots and predictive typing systems.


Methodology 
1. Data Collection 
A custom text file was used as the dataset for this project. The dataset contains multiple lines of text sentences, which serve as the training corpus for the model. The data was loaded, cleaned, and converted into a numerical format suitable for deep learning models. 

2. Data Preprocessing 
Preprocessing is crucial in NLP to transform raw text into numerical form. The main steps 
include: 
• Text Cleaning: Removing unnecessary punctuation, symbols, and converting text to lowercase. 
• Tokenization: Splitting text into tokens (individual words) and assigning each word a unique integer ID using Tokenizer from Keras. 
• Sequence Creation: Generating n-grams (word sequences) to train the model to predict the next word in a sequence. 
• Padding: Ensuring all sequences have the same length using pad_sequences() to make the input uniform. 
• One-hot Encoding: Converting categorical output labels into a binary matrix for training. 

3. Model Architecture 
The model was built using the Sequential API in TensorFlow/Keras with the following layers: 
• Embedding Layer: Converts input word indices into dense vector representations, capturing semantic relationships. 
• LSTM Layer: Learns contextual and temporal dependencies in the sequence. 
• Dense Layer: Fully connected layer with a softmax activation function to predict the probability of each word in the vocabulary. 
 
 The model was implemented and executed in Google Colab, taking advantage of GPU 
acceleration for faster training.
