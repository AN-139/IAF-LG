# IAF-LG: An Interactive Attention Fusion Network with Local and Global Perspective for Aspect-based Sentiment Analysis

This repository contains the implementation for the manuscript IAF-LG: An Interactive Attention Fusion Network with Local and Global Perspective for Aspect-based Sentiment Analysis

For any questions about the implementation, please email avidlearner139931@gmail.com

## Requirement
*Python version 3.6.7*

*Conda 4.7.12*

For creating the environment, all the necessary packages could be installed by running the *Env.yml* using the *Anaconda prompt*

## Directories
Place the datasets in the **Database** directory


Place the BERT base uncased model in the **BERT_base_uncased** directory 
(The pytorch version pre-trained bert-base-uncased model and vocabulary from the link provided by huggingface. Then change the value of parameter --bert_model_dir to the directory of the bert model)

## Preparations
For hyperparameter configurations, all the parameters can be set in the *researcher.py*

## Execution
Execute the file *Model.py* for training and testing on datasets
