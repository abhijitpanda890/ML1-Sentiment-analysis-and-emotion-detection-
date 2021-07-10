# ML1-Sentiment-analysis-and-emotion-detection-
This text classification model trains a Recurrent Neural Network (RNN) on the Yelp Polarity Reviews dataset for sentiment analysis.
Created by Zhang et al. in 2015, the Yelp Polarity Reviews Dataset contains 1,569,264 samples from the Yelp Dataset Challenge 2015. This subset has 280,000 training samples and 19,000 test samples in each polarity (positive and negative reviews). Dataset from FastAI's website., in the English language. Containing 1,569,264 in CSV file format.
Salient Features:

-	The raw text loaded by tfds needs to be processed before it can be used in a model. The simplest way to process text for training is using the experimental.preprocessing.TextVectorization layer. This layer will standardize, tokenize, and vectorize the data. Standardization refers to pre-processing the text, typically to remove punctuation or HTML elements to simplify the dataset. Tokenization refers to splitting strings into tokens (for example, splitting a sentence into individual words by splitting on whitespace). Vectorization refers to converting tokens into numbers so they can be fed into a neural network. 

-	The .prefetch() function is applied to overlap data pre-processing and model execution while training.

-	The layers are stacked sequentially to build the classifier:

I.	This model is built as a tf.keras.Sequential model.

II.	The first layer is the encoder, which converts the text to a sequence of token indices. 
After the encoder, there is an embedding layer. The Embedding layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index. When called, it converts the sequences of word indices to sequences of vectors. These vectors are trainable. After training (on enough data), words with similar meanings often have similar vectors. This index-lookup is much more efficient than the equivalent operation of passing a one-hot encoded vector through a tf.keras.layers.Dense layer.

III.	Then come the RNN layers.
A recurrent neural network (RNN) processes sequence input by iterating through the elements. RNNs pass the outputs from one timestep to their input on the next timestep. 
The tf.keras.layers.Bidirectional wrapper is used with two RNN layers. These layers propagate the input forward and backwards through the RNN and then concatenate the final output. 
The first layer has the return_sequences constructor argument that is set to true to return full sequences of successive outputs for each timestep (a 3D tensor of shape (batch size, timesteps, output features)).

IV.	Further there are two dense layers:

a.	The third last dense layer consists of 64 neuron units with a rectified linear unit (relu) activation.

b. The second output layer is a dropout layer that randomly drops out some fraction of a layer's input units every step of training, making it much harder for the network to learn those spurious patterns in the training data. Instead, it has to search for broad, general patterns, whose weight patterns tend to be more robust thereby countering overfitting.

c.	The last layer is densely connected with a single output node or neuron as we the model determine a single output, i.e., either positive or negative review.

After the RNN has converted the sequence to a single vector, the two layers.Dense do some final processing, and convert from this vector representation to a single logit as the classification output.

-	Since this is a binary classification problem and the model outputs single modified value (positive or negative), we use losses.BinaryCrossentropy loss function and adam as optimizer.

-	We then train the model for 10 epochs with a batch size of 128 at a time and 30 validation steps(batches) after each epoch.
After training the model, the following results are obtained:
loss: 0.1396, accuracy: 0.9422, f1_m: 0.9407, precision_m: 0.9613, recall_m: 0.9215, val_loss: 0.1554, val_accuracy: 0.9382, val_f1_m: 0.9383, val_precision_m: 0.9518, val_recall_m: 0.9256

-	Testing the model, on the test data we obtain the following results:
Test Loss: 0.14759980142116547
Test Accuracy: 0.9415000081062317
Test Precision: 0.9503315091133118
Test Recall: 0.9313362836837769
Test F1 Score: 0.9405220746994019

-Finally, we have saved the model under Keras to access it in future.
 


       
