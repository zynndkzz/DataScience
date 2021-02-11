# ReadMe
A hyperparameter is a parameter whose value is set before the learning process begins.Hyperparameters are not model parameters 
and they cannot be directly trained from the data. Because model parameters for ML algorithms are learned during training 
when we optimize a loss function using something like gradient descent.Given many models with different hyperparameters 
such as polynomial degrees , we can use a systematic approach -Hyperparameter tuning- to identify the 'best' function for a learning algorithm.

Let say we built a machine learning model and trained it on some data now questions are how to evaluate our model, will training model on more data 
improve its performance or do we need more features. In order to answer these questions,we can not train the model on the entire data set.
In addition to the sample data used to fit the model, we need a test set for  evaluation of our final model. But  evaluating each model while 
we are still tuning and building may cause data leakage.Therefore, we need to use a third unseen subset of the data -validation set-
optimize our model architecture. 

Our ultimate goal of machine learning models is to learn from examples and generalize some degree of knowledge that was gained while training.
When we run training algorithms on the data set,we decrease overall cost with more iteration but the case of line fit into all the points 
does not mean that we generalize the model. Maybe it will learn too much from training data.If the cost of the training set is very low but 
the cost of the validation set is very high this means our model is overfitting(high variance).On the other hand, it may not learn enough 
from training data.If the cost of the validation and training set is very high this means our model is underfitting(high bias).

So what is the right measure? ML models fulfill their purpose when generalized well. The performance is bounded by two undesirable outcomes -high variance
and high bias. Generally more desirable lies between overfitting and underfitting.  
