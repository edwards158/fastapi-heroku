# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Richard Edwards created the model.  It uses a Random Forest Classifier in scikit-learn to determine salary level.

## Intended Use
The model is used to determine salary level based on a handful of attributes.  The users are social studies researchers.   
## Training Data
The data was provided by udacity at:
- https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv
The data was cleaned and had 30162 rows.  A 80/20 split was used to seperate into train and test data.  To use the data for training a One Hot Encoder
was used on the features and a label binarizer was used on the labels.    
## Evaluation Data
The test was data used for evaluation and readings given for precision, recall and f1-score. 
## Metrics
Since the data set was unbalanced the f1-score was used to determine performance.  A f1-score 0.73 was achieved on the training data and 0.67 on the test data, indicating overfitting
## Ethical Considerations
Contains personal information
## Caveats and Recommendations
Only basic cleaning was carried out.  More advanced cleaning and feature selection would improve results.
