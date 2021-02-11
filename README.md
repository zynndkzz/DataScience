# Support Vector Machines
For SVM, our task is to find the best possible way to put a boundary between the two sets of points.
When a new point comes in, we can use this boundary to decide whether it belongs to class 1 or class 2.
A typical machine learning algorithm tries to find a boundary that divides the data in such a way that 
the misclassification error can be minimized.SVM differs from the other classification algorithms in the way that 
it chooses the decision boundary that maximizes the distance from the nearest data points of all the classes.An SVM doesn't 
merely find a decision boundary; it finds the most optimal decision boundary.The most optimal decision boundary is the one 
which has maximum margin from the nearest points of all the classes. The nearest points from the decision boundary that 
maximize the distance between the decision boundary and the points are called support vectors.

Two sets of points are said to be linearly separable if you can separate them using a single straight line.
However, not all data are linearly separable. In fact, in the real world, almost all the data are randomly distributed,
which makes it hard to separate different classes linearly.However, when there are more and more dimensions, 
computations within that space become more and more expensive. This is when the kernel trick comes in. It allows us to 
operate in the original feature space without computing the coordinates of the data in a higher dimensional space.In essence, 
what the kernel trick does for us is to offer a more efficient and less expensive way to transform data into higher dimensions. 
The kernel trick sounds like a “perfect” plan. However, one critical thing to keep in mind is that when we map data to a higher 
dimension, there are chances that we may overfit the model. Thus choosing the right kernel function (including the right parameters) 
and regularization are of great importance.

One of the kernel function is the  sigmoid functions.All sigmoid functions have the property that they map the entire number
line into a small range such as between 0 and 1, or -1 and 1, so one use of a sigmoid function is to convert a real value into
one that can be interpreted as a probability.Sigmoid functions are an important part of a logistic regression model. 
Logistic regression is a modification of linear regression for two-class classification, 
and converts one or more real-valued inputs into a probability.The final stage of a logistic regression model is often set to the logistic function,
which allows the model to output a probability.

Many datasets will not be linearly seperable.One way to cope with such datasets and still learn useful classifiers is to 
loosen some of the constraints by introducing slack variables.Slack variables are introduced to allow certain constraints 
to be violated.That is, certain training points will be allowed to within the margin.We want the number of points within the margin 
to be as small as possible,and of course we want their penetration of the margin to be as small as possible.To this end,we introduce one for each datapoint.
