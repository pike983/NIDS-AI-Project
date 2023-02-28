from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def runPCA(train_data):
    
    le = LabelEncoder()

    col_names = train_data.columns
    x_names = col_names[1:-2]
    y_names = col_names[-2:]
    X = train_data[x_names]
    y = train_data[y_names]

    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
    
    ## Running Predictions for attack_cat Category --------------------------------------------------------------
    sc = StandardScaler()
    train_features = sc.fit_transform(train_features)
    temp = train_labels.loc[:, 'attack_cat'].str.strip()
    train_labels_cat = le.fit_transform(temp.fillna('None'))
    train_labels_label = train_labels.loc[:, 'Label']
    
    test_features = sc.fit_transform(test_features)
    temp = test_labels.loc[:, 'attack_cat'].str.strip()
    test_labels_cat = temp.fillna('None')
    test_labels_label = test_labels.loc[:, 'Label']

    unique_categories = test_labels_cat.unique()
    unique_labels = test_labels_label.unique()

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
    
    ## Repeating Predictions with Dimensionality Reduced to 12 for Decision Tree Classifier --------------------------------------------------------------
    
    pca = PCA(n_components=12)
    tra_features = pca.fit_transform(train_features)
    tes_features = pca.transform(test_features)

    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(tra_features, train_labels_cat)
    y_pred = le.inverse_transform(dt_classifier.predict(tes_features))

    print("attack_cat")
    print("Classifier: Decision Tree Classifier")
    print(classification_report(test_labels_cat, y_pred, labels=unique_categories, zero_division=0))
    
    # Repeating Predictions with Dimensionality Reduced to 12 --------------------------------------------------------------

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

    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(tra_features, train_labels_label)
    y_pred = dt_classifier.predict(tes_features)

    print("Label")
    print("Classifier: Decision Tree Classifier")
    print(classification_report(test_labels_label, y_pred, target_names=unique_labels.astype('str'), zero_division=0))    

