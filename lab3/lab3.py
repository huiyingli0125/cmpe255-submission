import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        print(self.pima.head())
        self.X_test = None
        self.y_test = None

    def define_feature(self):
        #baseline
        #feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
        #solution1, solution3_1
        feature_cols = ['pregnant', 'glucose', 'bp', 'insulin', 'bmi', 'age']
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y

    def train(self):
        # split X and y into training and testing sets
        X, y = self.define_feature()
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)
        # train a logistic regression model on the training set
        #baseline
        logreg = LogisticRegression(class_weight='balanced')
        #solution2, solution3_2
        logreg = LogisticRegression(penalty = 'l2', C = 0.01, random_state = 0)
        logreg.fit(X_train, y_train)
        return logreg
    
    def predict(self):
        model = self.train()
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)
    
if __name__ == "__main__":
    classifer = DiabetesClassifier()
    result = classifer.predict()
    print(f"Predicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"score={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"confusion_matrix=${con_matrix}")
    
