import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer  # Import the imputer

class GreyWolfOptimize:
    def __init__(self, varRangesC, varRangesG):
        self.lowerBoundC = varRangesC[0]
        self.upperBoundC = varRangesC[1]
        self.lowerBoundG = varRangesG[0]
        self.upperBoundG = varRangesG[1]
    
    def runSVM(self, X, y):
        data = []
        for _ in range(5):
            c, g = self.initializingPop()
            clf = SVC(kernel='rbf', C=c, gamma=g)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            data.append({'C': c, 'gamma': g, 'accuracy': accuracy})
            
        return data
         
    def initializingPop(self):
        randomVariableDesignC = random.uniform(self.lowerBoundC, self.upperBoundC)
        randomVariableDesignG = random.uniform(self.lowerBoundG, self.upperBoundG)
        VariableDesignC = round(randomVariableDesignC, 2)
        VariableDesignG = round(randomVariableDesignG, 2)
    
        return VariableDesignC, VariableDesignG

    def mainGWO(self, X, y):
        GWO = GreyWolfOptimize(varRangesC, varRangesG)
        print(GWO.runSVM(X, y))

varRangesC = [0.01, 100]
varRangesG = [0.01, 50]

if __name__ == "__main__":
    # Read the dataset from a local CSV file
    dataset = pd.read_csv("C:\\Users\\WIN10\\OneDrive\\Documents\\Stuff\\UCP_Dataset1.csv", sep=';')
    X = dataset.drop(columns=['Real_Effort_Person_Hours', 'Project_No'])
    y = dataset['Real_Effort_Person_Hours']
    run = GreyWolfOptimize(varRangesC, varRangesG)
    run.mainGWO(X, y)
