import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

from DataParser import DataParser
from DataSplitter import DataSplitter

print("--- Reading data ---")
train_set = DataParser("Unified-Train-Set.csv")
test_set = DataParser("Unified-Test-Set.csv")
validation_set = DataParser("Unified-Validation-Set.csv")
train_set.label()
test_set.label()
validation_set.label()

# print("--- Seperating datasets ---")
# X_l_train, X_c_train = train_set.feature_select()
# X_l_test, X_c_test = test_set.feature_select()
# y_l_train, y_c_train = train_set.label_select()
# y_l_test, y_c_test = test_set.label_select()
# X_l_train = train_set.dataset_file[label_features]
# X_l_test = test_set.dataset_file[label_features]
# y_l_train = train_set.dataset_file['Label']
# y_l_test = test_set.dataset_file['Label']

# X_c_train = train_set.dataset_file[cat_features]
# X_c_test = test_set.dataset_file[cat_features]
# y_c_train = train_set.dataset_file['attack_cat']
# y_c_test = test_set.dataset_file['attack_cat']
x_names = train_set.dataset_file.columns[1:-2]
X_train = train_set.dataset_file[x_names]
y_l_train, y_c_train = train_set.label_select()
X_test = test_set.dataset_file[x_names]
y_l_test, y_c_test = test_set.label_select()
# print(x_names)
# print("--- Gaussian NB ---")
rfe = RFE(estimator=DecisionTreeClassifier(random_state=42),n_features_to_select=10)
# model = GaussianNB()
# pipeline = Pipeline(steps=[('s',rfe),('m',model)])

# print("--- Label prediction in progress ---")
# # y_l_pred = model.fit(X_l_train, y_l_train).predict(X_l_test)
# pipeline.fit(X_train, y_l_train)
# y_l_pred = pipeline.predict(X_test)

# print("--- Label prediction analysis ---")
# print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_l_test, y_l_pred)*100))
# print(metrics.classification_report(y_l_test, y_l_pred))

# print("--- Classification in progress ---")
# # y_c_pred = model.fit(X_c_train, y_c_train).predict(X_c_test)
# pipeline.fit(X_train, y_c_train)
# y_c_pred = pipeline.predict(X_test)

# print("--- Classification prediction analysis ---")
# print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_c_test, y_c_pred)*100))
# y_c_test1 = test_set.relabel(y_c_test)
# y_c_pred = test_set.relabel(y_c_pred)
# print(metrics.classification_report(y_c_test1, y_c_pred))

print("--- ADA Boost ---")
model = AdaBoostClassifier()
# print("--- Neural Network ---")
# model = MLPClassifier()
# print("--- Multinomial NB ---")
# model = MultinomialNB()
# print("--- QDA ---")
# model = QuadraticDiscriminantAnalysis()
print(model.get_params())
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
y_acc = 1.0
y_prev_acc = 0.0
step = 10
stop = 1

# while y_acc >= y_prev_acc:
#     print("--- RFE select: {:.2f} ---".format(step))
#     rfe.set_params(n_features_to_select=step)
#     pipeline.fit(X_train, y_c_train)
#     y_c_pred = pipeline.predict(X_test)
#     print("--- Prediction analysis ---")
#     y_acc = metrics.accuracy_score(y_c_test, y_c_pred)*100
#     print("Accuracy: {:.8f}%\n".format(y_acc))
#     print(metrics.classification_report(y_c_test, y_c_pred))
#     step += 1

# # Label training
pipeline.fit(X_train, y_l_train)
while  y_acc >= y_prev_acc and step <= 1e5:
    print("### RUN STEP {:.2f} ###".format(step))
    # for i in range(step, step*10, step):
    if y_acc < y_prev_acc:
        break
    y_prev_acc = y_acc
    i = step
    stop = i
    print("--- Label prediction in progress (alpha: {:.2f}) ---".format(i))
# y_l_pred = pipeline.predict(X_test)
    model.set_params(n_estimators=step)
    pipeline.fit(X_train, y_l_train)
    y_l_pred = pipeline.predict(X_test)

    print("--- Prediction analysis ---")
    y_acc = metrics.accuracy_score(y_l_test, y_l_pred)*100
    print("Accuracy: {:.8f}%\n".format(y_acc))
    print(metrics.classification_report(y_l_test, y_l_pred))
    if y_acc >= y_prev_acc:
        step *= 10
print("--- Solution found! (stop@: {:.2f} step: {:.2f}) ---".format(stop, step))
print("Accuracy: {:.2f}%\n".format(y_acc))
print(metrics.classification_report(y_l_test, y_l_pred))

# Category training
# pipeline.fit(X_train, y_c_train)
# print(model.get_params())
# step = 10
# while y_acc >= y_prev_acc and step <= 1e5:
#     print("### RUN STEP {:.8f} ###".format(step))
#     # i = step
#     # while i < step * 10:
#     # for i in range(step, step * 10, step):
#     i = step
#     if y_acc < y_prev_acc:
#         break
#     y_prev_acc = y_acc
#     stop = i
#     print("--- Classification prediction in progress (alpha: {:.8f}) ---".format(i))
# # y_c_pred = pipeline.predict(X_test)
#     model.set_params(n_estimators=step)
#     pipeline.fit(X_train, y_c_train)
#     y_c_pred = pipeline.predict(X_test)

#     print("--- Prediction analysis ---")
#     y_acc = metrics.accuracy_score(y_c_test, y_c_pred)*100
#     print("Accuracy: {:.8f}%\n".format(y_acc))
#     print(metrics.classification_report(y_c_test, y_c_pred))
#     if y_acc >= y_prev_acc:
#         step *= 10
# print("--- Solution found! (stop@: {:.8f} step: {:.8f}) ---".format(stop, step))
# print("Accuracy: {:.8f}%\n".format(y_acc))
# print(metrics.classification_report(y_c_test, y_c_pred))

    #     print("--- Classification in progress ---")
    #     # y_c_pred = model.fit(X_c_train, y_c_train).predict(X_c_test)
    #     pipeline.fit(X_train, y_c_train)
    #     y_c_pred = pipexline.predict(X_test)

    #     print("--- Classification prediction analysis ---")
    #     y_acc = metrics.accuracy_score(y_c_test, y_c_pred)*100
    #     print("Accuracy: {:.2f}%\n".format(y_acc))
    #     y_c_test2 = test_set.relabel(y_c_test)
    #     y_c_pred = test_set.relabel(y_c_pred)
    #     print(metrics.classification_report(y_c_test2, y_c_pred))
    # step *= 10