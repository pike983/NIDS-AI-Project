To work this project, navigate to the src folder in the terminal and run the following command:

``` python NIDS.py csvFilePath classificationMethod classificationGoal optionalPickledModelPath ```

The csvFilePath is the path to the csv file that contains the data to be classified.

The classificationMethod is the method to be used for classification.
	The three options for this argument are: "RFC" for Random Forest Classifier, "ADB" for AdaBoost, and "SVM" for Linear SVM.

The classificationTask is the task of the classification.
	The two options for this argument are: "Label" for classifying the label of the data, and "attack_cat" for classifying the attack type of the data.

The optionalPickledModelPath is the path to the pickled model to be used for classification.
	If this argument is not provided, the model will be trained from scratch.
	Otherwise this argument would be the path to the pickled model to be used for classification.
	These paths are: "RFC_Label_Model.pkl" for Random Forest Classifier for Label Classification, "ADB_Label_Model.pkl" for AdaBoost for Label Classification, "SVM_Label_Model.pkl" for Linear SVM for Label Classification, "RFC_Attack_Model.pkl" for Random Forest Classifier for Attack Classification, "ADB_Attack_Model.pkl" for AdaBoost for Attack Classification, and "SVM_Attack_Model.pkl" for Linear SVM for Attack Classification.

If this is not provided, the model will be trained from scratch.