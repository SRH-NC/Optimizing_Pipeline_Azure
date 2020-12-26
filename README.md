# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.

In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.

This model is then compared to an Azure AutoML run.

## Summary

This dataset contains bank marketing data that ultimately indicates in the last 'y' column if an individual "subscribes to a term deposit" (which I think means they opened a bank account).

Here is where I found descritions of the input variables - https://archive.ics.uci.edu/ml/datasets/bank+marketing

Our task is to build two models that best predict the outcome 'y' based on other customer data.  

One model uses logistic regression with parameters tuned by Hyperdrive and a second model with AutoML selecting the best model & parameters.
 
The best performing model was AutoML Voting Ensemble model with an accuracy rate of 91.684% accuracy.  

The Logistic Regression model tuned with Hyperdirve was close, resuting in an accuracy rate of 91.514%.

## Scikit-learn Pipeline

The pipeline architecture was implemented using two approachs.  

The First approach used a logistic regression model implemented in a python file.  A jupyter notebook file added the Hyperdrive components to run multiple logistic regression experiments tuned with diffferent hyperparameters.    

The second approach used AutoML to process the dataset and find the best outcome across multiple ML / Statistical models.  

I chose a basic logistic regression model using random parameter sampling with uniform distribution between 0 - 1 and various max iteration values between 10 and 150.  

In random sampling, hyperparameter values are randomly selected from the uniform search space.  This approach supports early termination of low-performance runs. 

The bandit policy terminates runs when the primary accuracy metric is not within the specified slack factor/slack amount compared to the best performing run.

In my project example, the early termination policy is applied at every interval when metrics are reported. Any run whose best metric is less than 1/(1+0.1) or 91% of the best performing run will be terminated.


## AutoML

The AutoML example completed approximately 35 child runs using various models and hyperarameters.  

As I mentioned above, the best performing model was the Voting Ensemble model with an accuracy of 91.684%.  

Other relevant output parameters generated from this Voting Ensemble run are AUC Macro (94.5%) and F1 Macro (76.8%).  

The AUC is good, but the F1 was lower and in looking at the confusion matrix I was able to dig deeper.  

In the confusion matrtix, Predicting 0 (didn't open an account) against true 0 was quite good at 96.6% correct vs 3.4% incorrect.  

The model was not as good at predicting 1 (would open an account) against true 1.  In this case it corrrectly predicted 52.5% vs incorrectly predicting 47.5%

Its not surprising that the Voting Ensemble model was the best performing.  This model combines the prediction from multiple other models.  

It is a technique that is often used to improve model performance, ideally achieivng better results than any single model in the ensemble.  

In addition to the best performing model I mentioned above, other ML models tested included:
* MaxAbsScaler & XGBoostClassifier (91.5% accuracy - 2nd best performing model), 
* SparseNormalizer & XGBoostClassifier (91.4% accuracy), 
* StandardScalerWrapper & RandomForest (89% accuracy). 

In the three examples above, the architecture includes a processing component to scal or normalize the data to help the algorighm perform well and prevent over / underfitting data.

One the data has been normalized, then an XGBoostClassifier (type of Gradient Boosting that combines decision trees) and RandomForest (a collection of decision trees with a single result) were the next best performing models.


## Pipeline comparison

At a practical level, the accuracy of the AutoML Voting Ensemble model and the logistic regression model perform essentialy the same.  

However several observations - the AutoML models include a wide spectrum of different model types that I was not aware of and would not typically include in test runs.  

The voting ensemble model also takes into account the performance of various models applied to my specific data set and selects the optimal set based on performance. 

Given that certain models perform better under different circumstances, this in the long run should improve model outcomes and reduce time to develop optimal models.

For example, gradient boosting algorithms can result in better performance than random forests.  However, gradient boosting may not be a good choice with lots of noise in data.

The power of AutoML and the Voting Ensemble is that it combines multiple models to improve overall predictions.  

Also, while it may be unique to my configuration when inspectng the individual child runs under experiments, they are generally fast, most completing in less than 1 minute.  




## Future work

For this first exercise, I used a basic logistic regression model.  

Future models can be improved adding / tuning hyperparameters. For example expand the space of the discrete hyperparemeters in hyperdrive config and leverage different sampeling methods (eg Grid).  


## Proof of cluster clean up

Due to the time it took to spin up enivronments using the lab provided environment, I created my own instance under my Azure account. 

I did delete the cluster and uploaded an image to my repo.
