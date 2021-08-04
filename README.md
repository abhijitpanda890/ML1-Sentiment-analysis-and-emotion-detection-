# ML1-Sentiment-analysis-and-emotion-detection-
This repository contains a text classification model that trains a Recurrent Neural Network (RNN) on the Yelp Polarity Reviews dataset for sentiment analysis.It is a project under the IITISoC programme(2021),  IIT Indore. 

Created by Zhang et al. in 2015, the Yelp Polarity Reviews Dataset contains 1,569,264 samples from the Yelp Dataset Challenge 2015. This subset has 280,000 training samples and 19,000 test samples in each polarity (positive and negative reviews). Dataset from FastAI's website., in the English language. Containing 1,569,264 in CSV file format.

 The repository contains three files:
 
 (i)Sentiment_Analysis(Yelp_Polarity_reviews).ipynb-the initial model(Kindly, refer to this first)
 
  After Training:
 
  | Metrics | Value |
  | --- | --- |
  | loss | 0.1396 |
  | accuracy | 0.9422 |
  | f1_m | 0.9407 | 
  | precision_m | 0.9613 |
  | recall_m | 0.9215 |
  | val_loss | 0.1554 |
  | val_accuracy | 0.9382 |
  | val_f1_m | 0.9383 |
  | val_precision_m | 0.9518 |
  | val_recall_m | 0.9256 |

  
   
 After Testing:
 
 | Metrics | Value |
 | --- | --- |
 | Test Loss | 0.14759980142116547 |
 | Test Accuracy | 0.9415000081062317 |
 | Test Precision | 0.9503315091133118 |
 | Test Recall | 0.9313362836837769 |
 | Test F1 Score | 0.9405220746994019 |
 
     
 
(ii)(Robust)Sentiment Analysis(Yelp-Polarity reviews).ipynb-This contains the model with overfitting overcome by using more dropout layers and earlystopping callback function.
 
 Metrics:
 
  After Training:
  
  | Metrics | Value |
  | --- | --- |
  | loss | 0.1766 |
  | accuracy | 0.9269 |
  | f1_m | 0.9248 | 
  | precision_m | 0.9482 |
  | recall_m | 0.9033 |
  | val_loss | 0.1791 |
  | val_accuracy | 0.9264 |
  | val_f1_m | 0.9268 |
  | val_precision_m | 0.9397 |
  | val_recall_m | 0.9147 |

  
  
  After Testing:
  
 | Metrics | Value |
 | --- | --- |
 | Test Loss | 0.17096687853336334 |
 | Test Accuracy | 0.93039470911026 |
 | Test Precision | 0.9406332969665527 |
 | Test Recall | 0.9187859892845154 |
 | Test F1 Score | 0.929294764995575 |
  

 (iii)Explanation-Sentiment Analysis.doc-An explanation of the various libraries and functions used to create, train and test the model.


Thank You

Group:ML-1

Sanjit Vyas        Abhijit Panda         Nilay Kushawaha





 


       
