import sys
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import DataParser

le = LabelEncoder()
k = 40
p = 1
weights = 'distance'
algo = 'auto'


def main():
    num_args = len(sys.argv)
    if num_args < 4: # Needs 4 args including the name
        print("Command Line Argument(s) Are Missing")
        return
    #print(sys.argv)
    
    # Import data
    #train_data = pd.read_csv('Unified-Train-Set.csv')
    #test_data = pd.read_csv('Unified-Test-Set.csv')
    
    # Parse data
    train_data = DataParser.DataParser('Unified-Train-Set.csv').dataset_file
    
    runPCA(train_data)

    #print("INFO")
    #print(data.info())
    
def runPCA(train_data):
    
    col_names = train_data.columns
    x_names = col_names[1:-2]
    y_names = col_names[-2:]
    X = train_data[x_names]
    y = train_data[y_names]

    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
    
    ## Running Predictions for attack_cat Category --------------------------------------------------------------
    #train_features = train_data.drop(['attack_cat', 'Label'], axis=1, inplace=False)
    sc = StandardScaler()
    train_features = sc.fit_transform(train_features)
    #temp = train_data.loc[:, 'attack_cat'].str.strip()
    #print("Training Features")
    #print(train_features)
    #print("Training Labels")
    #print(train_labels)
    #print("Test Features")
    #print(test_features)
    #print("Test Labels")
    #print(test_labels)
    #return
    temp = train_labels.loc[:, 'attack_cat'].str.strip()
    train_labels_cat = le.fit_transform(temp.fillna('None'))
    train_labels_label = train_labels.loc[:, 'Label']
    #print("TRAIN LABELS")
    #print(train_labels_cat)
    
    #test_features = test_data.drop(['attack_cat', 'Label'], axis=1, inplace=False)
    test_features = sc.fit_transform(test_features)
    #temp = test_data.loc[:, 'attack_cat'].str.strip()
    temp = test_labels.loc[:, 'attack_cat'].str.strip()
    test_labels_cat = temp.fillna('None')
    test_labels_label = test_labels.loc[:, 'Label']

    unique_categories = test_labels_cat.unique()
    unique_labels = test_labels_label.unique()

    #print(unique_categories)
    #print(unique_labels)

    # Plotting the Scree Plot ----------------------------------------------------------------------------------------------
    pca = PCA(n_components=None)

    tr_features = pca.fit_transform(train_features)

    te_features = pca.transform(test_features)

    feature_numbers = [i for i in range(1, 45)]

    plt.plot(feature_numbers, pca.explained_variance_ratio_, 'bo-')
    plt.xlabel("Principal Component")
    plt.ylabel("Proportion of Variance Explained")
    plt.title("Scree Plot")
    plt.show()
    
    #print("From the scree plot it appears that the elbow is at 4 principal components.")
    
    ##
    ## Code for KNClassifier and DTClassifier for attack_cat
    ##

    #pca = PCA(n_components=4)

    ## Reduce training set to 4 features
    #tr_features = pca.fit_transform(train_features)
    
    ## Reduce test set to 4 features
    #te_features = pca.transform(test_features)
    
    ##print("PCA Values, Params, and Covariance, and Components")
    ##print(pca.get_params())
    ##print(pca.get_covariance())
    ##print(pca.components_)

    #knn_classifier = KNeighborsClassifier(n_neighbors=k, p=p, weights=weights, algorithm=algo)
    #knn_classifier.fit(tr_features, train_labels_cat)
    #y_pred = le.inverse_transform(knn_classifier.predict(te_features))

    #print("attack_cat")
    #print("Classifier: K Neighbors Classifier")
    #print(classification_report(test_labels_cat, y_pred, labels=unique_categories, zero_division=0))
    
    ##results = []
    ##for result in range(len(y_pred)):
    ##    results.append(list(test_labels_cat)[result] == y_pred[result])
    ##error_rate = results.count(False)/len(results)
    ##print("4 PC Error rate: " + str(error_rate * 100) + "%")
    
    ## Repeating Predictions with Dimensionality Reduced to 4 for Decision Tree Classifier --------------------------------------------------------------

    pca = PCA(n_components=12)
    tra_features = pca.fit_transform(train_features)
    tes_features = pca.transform(test_features)

    #knn_classifier = KNeighborsClassifier(n_neighbors=5)
    #knn_classifier.fit(tra_features, train_labels)
    #y_pred = knn_classifier.predict(tes_features)
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(tra_features, train_labels_cat)
    y_pred = le.inverse_transform(dt_classifier.predict(tes_features))

    print("attack_cat")
    print("Classifier: Decision Tree Classifier")
    print(classification_report(test_labels_cat, y_pred, labels=unique_categories, zero_division=0))
    
    ##results = []
    ##for result in range(len(y_pred)):
    ##    results.append(list(test_labels_cat)[result] == y_pred[result])
    ##error_rate = results.count(False)/len(results)
    ##print("4 PC Error rate: " + str(error_rate * 100) + "%")
    
    ##
    ## Code for KNClassifier and DTClassifier for Label
    ##
    
    ### Running Predictions for Label Category --------------------------------------------------------------

    #pca = PCA(n_components=4)

    ## Reduce training set to 4 features
    #tr_features = pca.fit_transform(train_features)

    ## Reduce test set to 4 features
    #te_features = pca.transform(test_features)
    
    #knn_classifier = KNeighborsClassifier(n_neighbors=k, p=p, weights=weights, algorithm=algo)
    #knn_classifier.fit(tr_features, train_labels_label)
    #y_pred = knn_classifier.predict(te_features)
    ##dt_classifier = DecisionTreeClassifier()
    ##dt_classifier.fit(tr_features, train_labels_label)
    ##y_pred = dt_classifier.predict(te_features)

    #print("Label")
    #print("Classifier: K Neighbors Classifier")
    #print(classification_report(test_labels_label, y_pred, target_names=unique_labels.astype('str'), zero_division=0))
    
    #results = []
    #for result in range(len(y_pred)):
    #    results.append(list(test_labels_label)[result] == y_pred[result])
    #error_rate = results.count(False)/len(results)
    #print("4 PC Error rate: " + str(error_rate * 100) + "%")

    ## Plotting the Scree Plot ----------------------------------------------------------------------------------------------
    #pca = PCA(n_components=None)

    #tr_features = pca.fit_transform(train_features)

    #te_features = pca.transform(test_features)

    #feature_numbers = [i for i in range(1, 46)]

    ##plt.plot(feature_numbers, pca.explained_variance_ratio_, 'bo-')
    #plt.plot(feature_numbers, pca.explained_variance_, 'bo-')
    #plt.xlabel("Principal Component")
    #plt.ylabel("Proportion of Variance Explained")
    #plt.title("Scree Plot")
    #plt.show()
    
    #print("From the scree plot it appears that the elbow is at 4 principal components for Label.")
    
    # Repeating Predictions with Dimensionality Reduced to 4 --------------------------------------------------------------

    pca = PCA(n_components=12)
    tra_features = pca.fit_transform(train_features)
    tes_features = pca.transform(test_features)
    
    components = pca.components_
    for i in range(len(components)):
        print("Component " + str(i + 1))
        print(components[i])
        print("")
        
    eigenvalues = pca.explained_variance_
    for i in range(len(eigenvalues)):
        print("Eigenvalue " + str(i + 1))
        print(eigenvalues[i])
        print("")

    #knn_classifier = KNeighborsClassifier(n_neighbors=5)
    #knn_classifier.fit(tra_features, train_labels)
    #y_pred = knn_classifier.predict(tes_features)
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(tra_features, train_labels_label)
    y_pred = dt_classifier.predict(tes_features)

    print("Label")
    print("Classifier: Decision Tree Classifier")
    print(classification_report(test_labels_label, y_pred, target_names=unique_labels.astype('str'), zero_division=0))

    #results = []
    #for result in range(len(y_pred)):
    #    results.append(list(test_labels_label)[result] == y_pred[result])
    #error_rate = results.count(False)/len(results)
    #print("4 PC Error rate: " + str(error_rate * 100) + "%")

    
    
if __name__ == "__main__":
    main()