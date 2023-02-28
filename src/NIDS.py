import sys
from DataParser import DataParser
from RandomForest import runRFEandRFC


def main():
    args = sys.argv
    num_args = len(args)
    args.append(None)
    # Args:
    # 0: NIDS.py
    # 1: CSV file to predict from
    # 2: The Classification Method
    # 3: Classifying Label or attack_cat
    # 4: The pickled model file name
    if num_args < 4: # Needs 4 args including the name, 5th arg is pretrained model
        print("Command Line Argument(s) Are Missing")
        return
    pyFile = sys.argv[0]
    predFile = sys.argv[1]
    clMethod = sys.argv[2]
    task = sys.argv[3]
    pickledModel = None
    if num_args >= 4:
        pickledModel = sys.argv[4]
    else:
        pickledModel = None
    print("Python File: " + str(pyFile))
    print("Prediction File: " + str(predFile))
    print("Classification Method: " + str(clMethod))
    print("Classification Task: " + str(task))
    print("Pickled Model: " + str(pickledModel))
    
    data = DataParser(predFile)
    data.format()
    data.label()
    
    if clMethod == "RFC":
        print("Random Forest Classifier")
        print("Feature Selection: False")
        runRFEandRFC(data, task, pickledModel, False)
        print("Feature Selection: True")
        runRFEandRFC(data, task, pickledModel, True, True)
    elif clMethod == "ADB":
        print("AdaBoost Classifier")
    elif clMethod == "SVM":
        print("Support Vector Machine")
    else:
        print("Invalid Classification Method")
        return
    
if __name__ == "__main__":
    main()