# Logisitic_Regression

Classifying Iris dataset found at https://www.kaggle.com/uciml/iris with models in Python and R.

## R:

Performed K-nearest neighbors, linear discriminant analysis, and quadratic discriminant analysis on iris data

1) K-NN (KNN_iris.R)

2) LDA and QDA (LDA-QDA_iris.R)

Since binary classifier, performed log-reg out of curiosity to predict versicolor vs. setosa using Sepal.Width 

**RESULTS:**

All predictors -> LDA outperformed other methods
LDA: 96% accuracy, K-NN: 95.7% accuracy, QDA: 94.67% accuracy

Sepal.Width and Sepal.Length -> LDA and QDA performed identically
LDA: 81.33% accuracy, K-NN: 77.3% accuracy, QDA: 81.33% accuracy

Petal.Width and Petal.Length -> KNN and QDA performed identically 
LDA: 93.33% accuracy, K-NN: 97.33% accuracy, QDA: 97.33% accuracy

## Python:

Only performed log-reg in Python using Professor Ng's equations from Stanford's Machine Learning course. 
Thus only modelled versicolor vs. setosa using Sepal.Width (multiple preds couldn't converge)

1) Logic Approach (iris_Unvectorized.py)

2) Matrix Approach (iris_Vectorized.py)

## Technologies Used:
**Languages**:
Python, R

**Libraries**:
Pandas, Numpy, Matplotlib, scipy, caret
