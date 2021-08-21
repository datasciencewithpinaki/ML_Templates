# ML Template - Objective
Create an ML template that is reusable. This has the potential to save a lot of time for DS who can focus more on interpretation rather than working on something that can be automated.
<br> Learning around building ML Pipelines from scratch

# Pipeline
## Data Read 
Read the data from the db/ flat file
## EDA
Tables and charts to understand the data
## Preprocess
Preprocessing steps on the data before modeling includes steps like 
    - splitting data
    - missing value handling
    - outlier handling
    - encoding of catg features
    - scaling
    - dimension reduction
## ML Model Architecture
Setup Model(s)
### Model Training
Train model
### Model Evaluation
Predict on validation dataset
<br>Evaluate model on different metrics
### Voting or Ensemble
Use majority voting or an aggregate metric like average to get the ensemble prediction
## Saving Pipeline
Save pipeline of the flow so that new data can be passed throgh this pipeline for prediction
## Using preloaded model for predicting on new data
Test use cases
## Feedback mechanism
This is for getting user feedback to improve model further

## Cool links
few links here