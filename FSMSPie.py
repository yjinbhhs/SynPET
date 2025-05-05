# Databricks notebook source
pip list

# COMMAND ----------

# MAGIC %fs
# MAGIC ls s3://dl-dev-useast1-medical/bronze/BDH/MLAI

# COMMAND ----------

# MAGIC %fs
# MAGIC ls s3://dl-dev-useast1-medical/bronze/BDH/MLAI/inputfiles/

# COMMAND ----------

# MAGIC %sh
# MAGIC %pip uninstall xlrd
# MAGIC %pip install xlrd==1.2.0
# MAGIC %pip uninstall fsspec
# MAGIC pip install openpyxl
# MAGIC

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score, cross_val_predict,GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow import keras
import os
from shutil import copyfile

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC ls /dbfs/mnt/bdh_mlai_mnt/

# COMMAND ----------

#read the data and combine two lists together
case_df = pd.read_excel('/dbfs/mnt/bdh_mlai_mnt/yjin2/FSMsPie/RadReviewed001_Features_20211216.xlsx', sheet_name='1. Case-based Measures', engine='openpyxl')
lesion_df = pd.read_excel('/dbfs/mnt/bdh_mlai_mnt/yjin2/FSMsPie/RadReviewed001_Features_20211216.xlsx', sheet_name='2. Lesion-based Measures', engine='openpyxl')
#merge_df_orig = lesion_df.merge(case_df, on=['RadID', 'PatientID','StudyDate','PriorDate'])
merge_df_orig = lesion_df.merge(case_df, on=['PatientID','StudyDate','PriorDate'])
merge_df_orig.info()
random_seed = 88

# COMMAND ----------

# clean and organize the data

#merge_df_orig['Lesion_Classification'].value_counts()
#merge_df_orig['Lesion_Classification']=merge_df_orig['Lesion_Classification'].replace(['Tpos'],'TPos')
#merge_df_orig['Lesion_Classification']=merge_df_orig['Lesion_Classification'].replace(['Fpos'],'FPos')
merge_df_orig['Lesion_Classification'].value_counts()

# COMMAND ----------

merge_df_orig['Anatomical_Location'].value_counts()

# COMMAND ----------

#merge_df_orig['Proximity_to_Cortex'].value_counts()
#merge_df_orig['Proximity_to_Cortex']=merge_df_orig['Proximity_to_Cortex'].replace(['close '], 'close')
merge_df_orig['Proximity_to_Cortex'].value_counts()

# COMMAND ----------

merge_df_orig['Current_Scanner_Model'].value_counts()

# COMMAND ----------

merge_df_orig['Prior_Scanner_Model'].value_counts()

# COMMAND ----------

merge_df_orig['Scanner_Model_Change'] = (merge_df_orig['Current_Scanner_Model'] == merge_df_orig['Prior_Scanner_Model'])
merge_df_orig['Scanner_Model_Change'] = merge_df_orig['Scanner_Model_Change'].astype('int')
merge_df_orig['Scanner_Model_Change'].value_counts()

# COMMAND ----------

merge_df_orig.info()

# COMMAND ----------

#data cleaning - drop missing data
merge_df = merge_df_orig.dropna()
merge_df.info()

# COMMAND ----------

# transform the data
merge_df['Lesion_Volume_log'] = np.log(merge_df['Lesion_Volume'])
merge_df.drop(columns=['Lesion_Volume'], inplace=True)
merge_df.info()

# COMMAND ----------

#visualize the data
merge_df.hist(bins=50, figsize=(20,15))
plt.show()

# COMMAND ----------

#extract numerical and categorical attributes
merge_df_num = merge_df.drop(columns=['PatientID','StudyDate','PriorDate','LesionID','Lesion_Classification','Anatomical_Location','Proximity_to_Cortex','Current_Scanner_Model','Prior_Scanner_Model', 'Scanner_Model_Change'])
merge_df_cat = merge_df[['Anatomical_Location','Proximity_to_Cortex','Current_Scanner_Model','Prior_Scanner_Model', 'Scanner_Model_Change']]
merge_df_label = merge_df[['Lesion_Classification']]
merge_df_data = merge_df.drop(columns=['PatientID','StudyDate','PriorDate','LesionID','Lesion_Classification'])
cleanup_nums = {'Lesion_Classification': {'FPos': 1, 'TPos': 0}}
merge_df_label = merge_df_label.replace(cleanup_nums)
merge_df_label.value_counts()
merge_df_data.info()

# COMMAND ----------

#split training and test datasets
split=StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=random_seed)
for train, test in split.split(merge_df_data, merge_df_label):
    train_merge_df_data, test_merge_df_data = merge_df_data.iloc[train], merge_df_data.iloc[test]
    train_merge_df_num, test_merge_df_num = merge_df_num.iloc[train], merge_df_num.iloc[test]
    train_merge_df_cat, test_merge_df_cat = merge_df_cat.iloc[train], merge_df_cat.iloc[test]
    train_merge_df_label, test_merge_df_label = merge_df_label.iloc[train], merge_df_label.iloc[test]

# COMMAND ----------

train_merge_df_data['Anatomical_Location'].value_counts()

# COMMAND ----------

train_merge_df_data['Proximity_to_Cortex'].value_counts()

# COMMAND ----------

train_merge_df_data['Current_Scanner_Model'].value_counts()

# COMMAND ----------

train_merge_df_data['Prior_Scanner_Model'].value_counts()

# COMMAND ----------

#Normalize the numerical data and handle categorical attributes
num_attribs = list(merge_df_num)
cat_attribs_1 = ['Anatomical_Location','Proximity_to_Cortex', 'Scanner_Model_Change']
cat_attribs_2 = ['Current_Scanner_Model','Prior_Scanner_Model']
scanner_unique_cat = pd.unique(train_merge_df_data[cat_attribs_2].values.ravel())
full_pipeline = ColumnTransformer([
                                 ('num', StandardScaler(), num_attribs),
                                 ('cat_1', OneHotEncoder(handle_unknown='ignore'), cat_attribs_1),
                                 ('cat_2', OneHotEncoder(categories=[scanner_unique_cat]*2, handle_unknown='ignore'), cat_attribs_2)
                                ])

# COMMAND ----------

print(num_attribs)

# COMMAND ----------

#prepare the dataset
train_merge_df_data_prepared = full_pipeline.fit_transform(train_merge_df_data)
train_merge_df_label_prepared = train_merge_df_label.to_numpy().ravel()
print(train_merge_df_data_prepared.shape)
test_merge_df_data_prepared = full_pipeline.transform(test_merge_df_data)
test_merge_df_label_prepared = test_merge_df_label.to_numpy().ravel()
print(test_merge_df_data_prepared.shape)


# COMMAND ----------

print(test_merge_df_data_prepared[0,:])

# COMMAND ----------

#visualize the data
pd.DataFrame(train_merge_df_data_prepared).hist(bins=50, figsize=(20,15))
plt.show()

# COMMAND ----------

# mdoel hyperparameter selection
log_clf = LogisticRegression(C=5, max_iter=10000, solver='saga', random_state=random_seed)
svc_clf = SVC(C=2, kernel='linear', probability=True, random_state=random_seed)
rnd_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_seed)

#kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

"""
voting_clf=VotingClassifier(
    estimators=[('lr',log_clf), ('rf',rnd_clf), ('svc',svc_clf)],
    voting='hard')
param_grid={'lr__C': [1],
            'svc__C': [1],
            'svc__kernel': ['rbf'],
            'rf__n_estimators': [10, 50, 100, 500],
            'rf__max_depth': [10, 20]
             }

grid = GridSearchCV(estimator=voting_clf, param_grid=param_grid, cv=5)
    
for train_index, test_index in kf.split(merge_df_data_prepared, merge_df_label_prepared):
    X_train, X_test = merge_df_data_prepared[train_index], merge_df_data_prepared[test_index]
    y_train, y_test = merge_df_label_prepared[train_index], merge_df_label_prepared[test_index]
    grid.fit(X_train,y_train)
    print (grid.best_params_)
    y_pred=grid.predict(X_test)
    print(classification_report(y_test, y_pred))
"""

# COMMAND ----------

# model training
log_clf.fit(train_merge_df_data_prepared, train_merge_df_label_prepared)
svc_clf.fit(train_merge_df_data_prepared, train_merge_df_label_prepared)
rnd_clf.fit(train_merge_df_data_prepared, train_merge_df_label_prepared)

# model prediction
log_pred = log_clf.predict(test_merge_df_data_prepared)
log_scores = log_clf.decision_function(test_merge_df_data_prepared)
svc_pred = svc_clf.predict(test_merge_df_data_prepared)
svc_scores = svc_clf.decision_function(test_merge_df_data_prepared)
rnd_pred = rnd_clf.predict(test_merge_df_data_prepared)
rnd_scores = rnd_clf.predict_proba(test_merge_df_data_prepared)

# COMMAND ----------

# Logistic Regression Performance Measures
conf_matrx = confusion_matrix(test_merge_df_label_prepared, log_pred)
print(conf_matrx)
log_acc = (conf_matrx[0,0] + conf_matrx[1,1]) / np.sum(conf_matrx)
log_precision = precision_score(test_merge_df_label_prepared, log_pred)
log_recall = recall_score(test_merge_df_label_prepared, log_pred)
log_f1 = f1_score(test_merge_df_label_prepared, log_pred)
print("Logistic Regression accuracy is: {}".format(np.round(log_acc, 4)))
print("Logistic Regression precision is: {}".format(np.round(log_precision, 4)))
print("Logistic Regression recall is: {}".format(np.round(log_recall, 4)))
print("Logistic Regression F1 score is: {}".format(np.round(log_f1, 4)))

# COMMAND ----------

# SVM Performance Measures
conf_matrx = confusion_matrix(test_merge_df_label_prepared, svc_pred)
print(conf_matrx)
svc_acc = (conf_matrx[0,0] + conf_matrx[1,1]) / np.sum(conf_matrx)
svc_precision = precision_score(test_merge_df_label_prepared, svc_pred)
svc_recall = recall_score(test_merge_df_label_prepared, svc_pred)
svc_f1 = f1_score(test_merge_df_label_prepared, svc_pred)
print("SVM accuracy is: {}".format(np.round(svc_acc, 4)))
print("SVM precision is: {}".format(np.round(svc_precision, 4)))
print("SVM recall is: {}".format(np.round(svc_recall, 4)))
print("SVM F1 score is: {}".format(np.round(svc_f1, 4)))

# COMMAND ----------

# Random Forest Performance Measures
conf_matrx = confusion_matrix(test_merge_df_label_prepared, rnd_pred)
print(conf_matrx)
rnd_acc = (conf_matrx[0,0] + conf_matrx[1,1]) / np.sum(conf_matrx)
rnd_precision = precision_score(test_merge_df_label_prepared, rnd_pred)
rnd_recall = recall_score(test_merge_df_label_prepared, rnd_pred)
rnd_f1 = f1_score(test_merge_df_label_prepared, rnd_pred)
print("Random Forest accuracy is: {}".format(np.round(rnd_acc, 4)))
print("Random Forest precision is: {}".format(np.round(rnd_precision, 4)))
print("Random Forest recall is: {}".format(np.round(rnd_recall, 4)))
print("Random Forest F1 score is: {}".format(np.round(rnd_f1, 4)))

ft_rank = rnd_clf.feature_importances_
ft_list = num_attribs + cat_attribs_1 + cat_attribs_2
ft_freq = {}
ft_count = 0

for ft in ft_list:
    if num_attribs.count(ft):
        ft_freq[ft] = ft_rank[ft_count]
        ft_count += 1
    elif cat_attribs_1.count(ft):
        uniq_value_count = len(pd.unique(train_merge_df_data[ft].values.ravel()))
        ft_freq[ft] = np.sum(ft_rank[ft_count:ft_count+uniq_value_count])
        ft_count += uniq_value_count
    else:
        uniq_value_count = len(pd.unique(train_merge_df_data[cat_attribs_2].values.ravel()))
        ft_freq[ft] = np.sum(ft_rank[ft_count:ft_count+uniq_value_count])
        ft_count += uniq_value_count        

ft_freq = dict(sorted(ft_freq.items(), key=lambda item: item[1], reverse=True))        
print(ft_freq)
values = ft_freq.values()
total = sum(values)
print(total)

# COMMAND ----------

# model training with cross validation
#np.set_printoptions(precision=3)
#cv_score_log = cross_val_score(log_clf, merge_df_data_prepared, merge_df_label_prepared.ravel(), cv= kf, scoring="accuracy")
#print("Logistic Regression accuracies for each fold are: {}".format(cv_score_log))
#mean_score_log=np.mean(cv_score_log)
#std_score_log=np.std(cv_score_log)
#print("Logistic Regression accuracy is: {:.2f} +/- {:.2f}".format(mean_score_log, std_score_log))

#cv_score_svc = cross_val_score(svc_clf, merge_df_data_prepared, merge_df_label_prepared.ravel(), cv= kf, scoring="accuracy")
#print("SVM accuracies for each fold are: {}".format(cv_score_svc))
#mean_score_svc=np.mean(cv_score_svc)
#std_score_svc=np.std(cv_score_svc)
#print("SVM accuracy is: {:.2f} +/- {:.2f}".format(mean_score_svc, std_score_svc))

#cv_score_rnd = cross_val_score(rnd_clf, merge_df_data_prepared, merge_df_label_prepared.ravel(), cv= kf, scoring="accuracy")
#print("Random Forest accuracies for each fold are: {}".format(cv_score_rnd))
#mean_score_rnd=np.mean(cv_score_rnd)
#std_score_rnd=np.std(cv_score_rnd)
#print("Random Forest mean accuracy is: {:.2f} +/- {:.2f}".format(mean_score_rnd, std_score_rnd))

#pred_log_proba = cross_val_predict(log_clf, merge_df_data_prepared, merge_df_label_prepared, cv= kf, method="predict_proba")
#pred_svc_proba = cross_val_predict(svc_clf, merge_df_data_prepared, merge_df_label_prepared, cv= kf, method="predict_proba")
#pred_rnd_proba = cross_val_predict(rnd_clf, merge_df_data_prepared, merge_df_label_prepared, cv= kf, method="predict_proba")

# COMMAND ----------

# print results
auc_log = roc_auc_score(test_merge_df_label_prepared, log_scores)
print("Logistic Regression AUC is: {:.4f}".format(auc_log))
auc_svc = roc_auc_score(test_merge_df_label_prepared, svc_scores)
print("SVM AUC is: {:.4f}".format(auc_svc))
auc_rnd = roc_auc_score(test_merge_df_label_prepared, rnd_scores[:,1])
print("Random Forest AUC is: {:.4f}".format(auc_rnd))   
    
fpr_log, tpr_log, thresholds_log = roc_curve(test_merge_df_label_prepared, log_scores)
fpr_svc, tpr_svc, thresholds_svc = roc_curve(test_merge_df_label_prepared, svc_scores)
fpr_rnd, tpr_rnd, thresholds_rnd = roc_curve(test_merge_df_label_prepared, rnd_scores[:,1])

# COMMAND ----------

# train a deep learning network


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"
print([tf.__version__, tf.config.list_physical_devices('GPU')])

#cv_score_dnn=[]
input_shape = train_merge_df_data_prepared.shape[1]

def build_model():
    model=keras.models.Sequential([
          keras.Input(shape=(input_shape,)),
          keras.layers.Dense(300, kernel_initializer="glorot_uniform"),
          keras.layers.BatchNormalization(),
          keras.layers.Activation("relu"),
          keras.layers.Dropout(0.5),
      
          keras.layers.Dense(200, kernel_initializer="glorot_uniform"),
          keras.layers.BatchNormalization(),
          keras.layers.Activation("relu"),
          keras.layers.Dropout(0.5),

          keras.layers.Dense(100, kernel_initializer="glorot_uniform"),
          keras.layers.BatchNormalization(),
          keras.layers.Activation("relu"),
          keras.layers.Dropout(0.5),
          
          keras.layers.Dense(50, kernel_initializer="glorot_uniform"),
          keras.layers.BatchNormalization(),
          keras.layers.Activation("relu"),
          keras.layers.Dropout(0.5),
            
          keras.layers.Dense(10, kernel_initializer="glorot_uniform"),
          keras.layers.BatchNormalization(),
          keras.layers.Activation("relu"),
          keras.layers.Dropout(0.5),      
            
          keras.layers.Dense(1, kernel_initializer="glorot_uniform"),
          keras.layers.BatchNormalization(),
          keras.layers.Activation("sigmoid")
          ])

    model.summary()

    learning_rate=0.0001
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
 
    model.compile(loss="binary_crossentropy", 
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model

keras_model = build_model()

   
history = keras_model.fit(train_merge_df_data_prepared, train_merge_df_label_prepared, epochs=500, batch_size=10)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

dnn_acc = keras_model.evaluate(test_merge_df_data_prepared, test_merge_df_label_prepared)
dnn_scores = keras_model.predict(test_merge_df_data_prepared).ravel() 
print("%s: %.2f%%" % (keras_model.metrics_names[0], dnn_acc[0]*100))
print("%s: %.2f%%" % (keras_model.metrics_names[1], dnn_acc[1]*100))

dnn_bin_scores = np.round(dnn_scores)
conf_matrx = confusion_matrix(test_merge_df_label_prepared, dnn_bin_scores)
print(conf_matrx)
dnn_acc = (conf_matrx[0,0] + conf_matrx[1,1]) / np.sum(conf_matrx)
dnn_precision = precision_score(test_merge_df_label_prepared, dnn_bin_scores)
dnn_recall = recall_score(test_merge_df_label_prepared, dnn_bin_scores)
dnn_f1 = f1_score(test_merge_df_label_prepared, dnn_bin_scores)
print("Deep Neural Network accuracy is: {}".format(np.round(dnn_acc, 4)))
print("Deep Neural Network precision is: {}".format(np.round(dnn_precision, 4)))
print("Deep Neural Network recall is: {}".format(np.round(dnn_recall, 4)))
print("Deep Neural Network F1 score is: {}".format(np.round(dnn_f1, 4)))

keras_model.save("dnn_model.h5")
copyfile("dnn_model.h5", "/dbfs/mnt/bdh_mlai_mnt/yjin2/FSMsPie/dnn_model.h5")

#pred_dnn_proba = cross_val_predict(keras_model, merge_df_data_prepared, merge_df_label_prepared, cv= kf, method="predict_proba")
auc_dnn = roc_auc_score(test_merge_df_label_prepared, dnn_scores)
print("DNN AUC is: {:.4f}".format(auc_dnn))
fpr_dnn, tpr_dnn, thresholds_dnn = roc_curve(test_merge_df_label_prepared, dnn_scores)


# COMMAND ----------

plt.plot(fpr_log, tpr_log, "r-", label='Logistic Regression, AUC={:.2f}'.format(auc_log))
plt.plot(fpr_svc, tpr_svc, "b-", label='SVM, AUC={:.2f}'.format(auc_svc))
plt.plot(fpr_rnd, tpr_rnd, "y-", label='Random Forest, AUC={:.2f}'.format(auc_rnd))
plt.plot(fpr_dnn, tpr_dnn, "k-", label='DNN, AUC={:.2f}'.format(auc_dnn))
plt.plot([0,1],[0,1],'k--')
plt.legend(loc="lower right")
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate(Recall)")
plt.show()

cor = dnn_bin_scores-test_merge_df_label_prepared
cor = pd.DataFrame(cor, columns=['Correctness'], index=test_merge_df_data.index).astype(float)
cor['Correctness'].replace([1, -1], "Wrong", inplace=True)
cor['Correctness'].replace(0, "Correct", inplace=True)
results = pd.DataFrame(dnn_scores,columns=['False Positive Probability'], index=test_merge_df_data.index)
results['True Positive Probability'] = 1-results['False Positive Probability']
results = pd.concat([results, cor],axis=1)
results = merge_df_orig.merge(results, left_index=True, right_index=True, how='left')

print(results)
results.to_csv('/dbfs/mnt/bdh_mlai_mnt/yjin2/FSMsPie/results.csv', index=False)

# COMMAND ----------

results['Correctness'].value_counts()

# COMMAND ----------

#visualize the separability of the results
bins = np.linspace(0, 1, 50)

FPos_Prob=results.loc[results['Lesion_Classification']=='FPos','False Positive Probability']
TPos_Prob=results.loc[results['Lesion_Classification']=='TPos','False Positive Probability']
plt.hist(FPos_Prob, bins, color='green', alpha=0.5, label='False Positive')
plt.hist(TPos_Prob, bins, color='red', alpha=0.5, label='True Positive')
plt.legend(loc='upper center')
plt.xlabel("False Positive Probability")
plt.show()

# COMMAND ----------

#read the data and combine two lists together
case_df_test1 = pd.read_excel('/dbfs/mnt/bdh_mlai_mnt/yjin2/FSMsPie/888MS007_Endpoints20210415_AT20210819.xlsx', sheet_name='1. Case-based Measures', engine='openpyxl')
lesion_df_test1 = pd.read_excel('/dbfs/mnt/bdh_mlai_mnt/yjin2/FSMsPie/888MS007_Endpoints20210415_AT20210819.xlsx', sheet_name='2. Lesion-based Measures', engine='openpyxl')
merge_df_test1_orig = lesion_df_test1.merge(case_df_test1, on=['RadID', 'PatientID','StudyDate','PriorDate'])
merge_df_test1_orig.drop(columns='RadID', inplace=True)
merge_df_test1_orig.info()

# COMMAND ----------

merge_df_test1_orig['Lesion_Classification'].value_counts()

# COMMAND ----------

# clean and organize the data

merge_df_test1_orig['Lesion_Classification']=merge_df_test1_orig['Lesion_Classification'].replace(['Tpos'],'TPos')
merge_df_test1_orig['Lesion_Classification']=merge_df_test1_orig['Lesion_Classification'].replace(['Fpos'],'FPos')
merge_df_test1_orig['Lesion_Classification'].value_counts()

# COMMAND ----------

# clean and organize the data
merge_df_test1_orig['Anatomical_Location'].value_counts()

# COMMAND ----------

merge_df_test1_orig['Proximity_to_Cortex'].value_counts()

# COMMAND ----------

merge_df_test1_orig['Proximity_to_Cortex'] = merge_df_test1_orig['Proximity_to_Cortex'].replace(['close '], 'close')
merge_df_test1_orig['Proximity_to_Cortex'].value_counts()

# COMMAND ----------

merge_df_test1_orig['Current_Scanner_Model'].value_counts()

# COMMAND ----------

merge_df_test1_orig['Scanner_Model_Change'] = (merge_df_test1_orig['Current_Scanner_Model'] == merge_df_test1_orig['Prior_Scanner_Model'])
merge_df_test1_orig['Scanner_Model_Change'] = merge_df_test1_orig['Scanner_Model_Change'].astype('int')
merge_df_test1_orig['Scanner_Model_Change'].value_counts()

# COMMAND ----------

merge_df_test1_orig.info()

# COMMAND ----------

#data cleaning - drop missing data
merge_df_test1 = merge_df_test1_orig.dropna()
merge_df_test1.info()

# COMMAND ----------

merge_df_test1['Lesion_Classification'].value_counts()

# COMMAND ----------

# transform the data
merge_df_test1['Lesion_Volume_log'] = np.log(merge_df_test1['Lesion_Volume'])
merge_df_test1.drop(columns=['Lesion_Volume'], inplace=True)
merge_df_test1.info()

# COMMAND ----------

#visualize the data
merge_df_test1.hist(bins=50, figsize=(20,15))
plt.show()

# COMMAND ----------

#extract numerical and categorical attributes
merge_df_label_test1 = merge_df_test1[['Lesion_Classification']]
merge_df_data_test1 = merge_df_test1.drop(columns=['PatientID','StudyDate','PriorDate','LesionID','Lesion_Classification'])
cleanup_nums={'Lesion_Classification': {'FPos': 1, 'TPos': 0}}
merge_df_label_test1 = merge_df_label_test1.replace(cleanup_nums)
merge_df_label_test1.value_counts()
merge_df_data_test1.info()

# COMMAND ----------

merge_df_data_test1_prepared = full_pipeline.transform(merge_df_data_test1)
merge_df_label_test1_prepared = merge_df_label_test1.to_numpy().ravel()
print(merge_df_data_test1_prepared.shape)

# COMMAND ----------

#visualize the data
pd.DataFrame(merge_df_data_test1_prepared).hist(bins=50, figsize=(20,15))
plt.show()

# COMMAND ----------

# model prediction
log_pred_test1 = log_clf.predict(merge_df_data_test1_prepared)
log_scores_test1 = log_clf.decision_function(merge_df_data_test1_prepared)
svc_pred_test1 = svc_clf.predict(merge_df_data_test1_prepared)
svc_scores_test1 = svc_clf.decision_function(merge_df_data_test1_prepared)
rnd_pred_test1 = rnd_clf.predict(merge_df_data_test1_prepared)
rnd_scores_test1 = rnd_clf.predict_proba(merge_df_data_test1_prepared)

# COMMAND ----------

# Logistic Regression Performance Measures
conf_matrx_test1 = confusion_matrix(merge_df_label_test1_prepared, log_pred_test1)
print(conf_matrx_test1)
log_acc_test1 = (conf_matrx_test1[0,0] + conf_matrx_test1[1,1]) / np.sum(conf_matrx_test1)
log_precision_test1 = precision_score(merge_df_label_test1_prepared, log_pred_test1)
log_recall_test1 = recall_score(merge_df_label_test1_prepared, log_pred_test1)
log_f1_test1 = f1_score(merge_df_label_test1_prepared, log_pred_test1)
print("Logistic Regression accuracy is: {}".format(np.round(log_acc_test1, 4)))
print("Logistic Regression precision is: {}".format(np.round(log_precision_test1, 4)))
print("Logistic Regression recall is: {}".format(np.round(log_recall_test1, 4)))
print("Logistic Regression F1 score is: {}".format(np.round(log_f1_test1, 4)))

# COMMAND ----------

# SVM Performance Measures
conf_matrx_test1 = confusion_matrix(merge_df_label_test1_prepared, svc_pred_test1)
print(conf_matrx_test1)
svc_acc_test1 = (conf_matrx_test1[0,0] + conf_matrx_test1[1,1]) / np.sum(conf_matrx_test1)
svc_precision_test1 = precision_score(merge_df_label_test1_prepared, svc_pred_test1)
svc_recall_test1 = recall_score(merge_df_label_test1_prepared, svc_pred_test1)
svc_f1_test1 = f1_score(merge_df_label_test1_prepared, svc_pred_test1)
print("SVM accuracy is: {}".format(np.round(svc_acc_test1, 4)))
print("SVM precision is: {}".format(np.round(svc_precision_test1, 4)))
print("SVM recall is: {}".format(np.round(svc_recall_test1, 4)))
print("SVM F1 score is: {}".format(np.round(svc_f1_test1, 4)))

# COMMAND ----------

# Random Forest Performance Measures
conf_matrx_test1 = confusion_matrix(merge_df_label_test1_prepared, rnd_pred_test1)
print(conf_matrx_test1)
rnd_acc_test1 = (conf_matrx_test1[0,0] + conf_matrx_test1[1,1]) / np.sum(conf_matrx_test1)
rnd_precision_test1 = precision_score(merge_df_label_test1_prepared, rnd_pred_test1)
rnd_recall_test1 = recall_score(merge_df_label_test1_prepared, rnd_pred_test1)
rnd_f1_test1 = f1_score(merge_df_label_test1_prepared, rnd_pred_test1)
print("Random Forest accuracy is: {}".format(np.round(rnd_acc_test1, 4)))
print("Random Forest precision is: {}".format(np.round(rnd_precision_test1, 4)))
print("Random Forest recall is: {}".format(np.round(rnd_recall_test1, 4)))
print("Random Forest F1 score is: {}".format(np.round(rnd_f1_test1, 4)))

# COMMAND ----------

# print results
auc_log_test1 = roc_auc_score(merge_df_label_test1_prepared, log_scores_test1)
print("Logistic Regression AUC is: {:.4f}".format(auc_log_test1))
auc_svc_test1 = roc_auc_score(merge_df_label_test1_prepared, svc_scores_test1)
print("SVM AUC is: {:.4f}".format(auc_svc_test1))
auc_rnd_test1 = roc_auc_score(merge_df_label_test1_prepared, rnd_scores_test1[:,1])
print("Random Forest AUC is: {:.4f}".format(auc_rnd_test1))   
    
fpr_log_test1, tpr_log_test1, thresholds_log_test1 = roc_curve(merge_df_label_test1_prepared, log_scores_test1)
fpr_svc_test1, tpr_svc_test1, thresholds_svc_test1 = roc_curve(merge_df_label_test1_prepared, svc_scores_test1)
fpr_rnd_test1, tpr_rnd_test1, thresholds_rnd_test1 = roc_curve(merge_df_label_test1_prepared, rnd_scores_test1[:,1])

# COMMAND ----------

keras_model=keras.models.load_model("/dbfs/mnt/bdh_mlai_mnt/yjin2/FSMsPie/dnn_model.h5")

dnn_acc_test1 = keras_model.evaluate(merge_df_data_test1_prepared, merge_df_label_test1_prepared)
dnn_scores_test1 = keras_model.predict(merge_df_data_test1_prepared).ravel()
print("%s: %.2f%%" % (keras_model.metrics_names[0], dnn_acc_test1[0]*100))
print("%s: %.2f%%" % (keras_model.metrics_names[1], dnn_acc_test1[1]*100))

dnn_bin_scores_test1 = np.round(dnn_scores_test1)
conf_matrx_test1 = confusion_matrix(merge_df_label_test1_prepared, dnn_bin_scores_test1)
print(conf_matrx_test1)
dnn_acc_test1 = (conf_matrx_test1[0,0] + conf_matrx_test1[1,1]) / np.sum(conf_matrx_test1)
dnn_precision_test1 = precision_score(merge_df_label_test1_prepared, dnn_bin_scores_test1)
dnn_recall_test1 = recall_score(merge_df_label_test1_prepared, dnn_bin_scores_test1)
dnn_f1_test1 = f1_score(merge_df_label_test1_prepared, dnn_bin_scores_test1)
print("Deep Neural Network accuracy is: {}".format(np.round(dnn_acc_test1, 4)))
print("Deep Neural Network precision is: {}".format(np.round(dnn_precision_test1, 4)))
print("Deep Neural Network recall is: {}".format(np.round(dnn_recall_test1, 4)))
print("Deep Neural Network F1 score is: {}".format(np.round(dnn_f1_test1, 4)))


auc_dnn_test1 = roc_auc_score(merge_df_label_test1_prepared, dnn_scores_test1)
print("DNN AUC is: {:.4f}".format(auc_dnn_test1))

# COMMAND ----------

#read the data and combine two lists together
case_df_test2 = pd.read_excel('/dbfs/mnt/bdh_mlai_mnt/yjin2/FSMsPie/LesionGS_Features_20210702_AT20210831.xlsx', sheet_name='1. Case-based Measures', engine='openpyxl')
lesion_df_test2 = pd.read_excel('/dbfs/mnt/bdh_mlai_mnt/yjin2/FSMsPie/LesionGS_Features_20210702_AT20210831.xlsx', sheet_name='2. Lesion-based Measures', engine='openpyxl')
merge_df_test2_orig = lesion_df_test2.merge(case_df_test2, on=['PatientID','StudyDate','PriorDate'])
merge_df_test2_orig.info()

# COMMAND ----------

merge_df_test2_orig['Lesion_Classification'].value_counts()

# COMMAND ----------

# clean and organize the data

merge_df_test2_orig['Lesion_Classification']=merge_df_test2_orig['Lesion_Classification'].replace(['Tpos'],'TPos')
merge_df_test2_orig['Lesion_Classification']=merge_df_test2_orig['Lesion_Classification'].replace(['Fpos'],'FPos')
merge_df_test2_orig['Lesion_Classification'].value_counts()

# COMMAND ----------

# clean and organize the data
merge_df_test2_orig['Anatomical_Location'].value_counts()

# COMMAND ----------

merge_df_test2_orig['Proximity_to_Cortex'].value_counts()

# COMMAND ----------

merge_df_test2_orig['Current_Scanner_Model'].value_counts()

# COMMAND ----------

merge_df_test2_orig['Scanner_Model_Change'] = (merge_df_test2_orig['Current_Scanner_Model'] == merge_df_test2_orig['Prior_Scanner_Model'])
merge_df_test2_orig['Scanner_Model_Change'] = merge_df_test2_orig['Scanner_Model_Change'].astype('int')
merge_df_test2_orig['Scanner_Model_Change'].value_counts()

# COMMAND ----------

merge_df_test2_orig.info()

# COMMAND ----------

#data cleaning - drop missing data
merge_df_test2 = merge_df_test2_orig.dropna()
merge_df_test2.info()

# COMMAND ----------

# transform the data
merge_df_test2['Lesion_Volume_log'] = np.log(merge_df_test2['Lesion_Volume'])
merge_df_test2.drop(columns=['Lesion_Volume'], inplace=True)
merge_df_test2.info()

# COMMAND ----------

merge_df_test2['Lesion_Classification'].value_counts()

# COMMAND ----------

#visualize the data
merge_df_test2.hist(bins=50, figsize=(20,15))
plt.show()

# COMMAND ----------

#extract numerical and categorical attributes
merge_df_label_test2 = merge_df_test2[['Lesion_Classification']]
merge_df_data_test2 = merge_df_test2.drop(columns=['PatientID','StudyDate','PriorDate','LesionID','Lesion_Classification'])
cleanup_nums={'Lesion_Classification': {'FPos': 1, 'TPos': 0}}
merge_df_label_test2 = merge_df_label_test2.replace(cleanup_nums)
merge_df_label_test2.value_counts()
merge_df_data_test2.info()

# COMMAND ----------

merge_df_data_test2_prepared = full_pipeline.transform(merge_df_data_test2)
merge_df_label_test2_prepared = merge_df_label_test2.to_numpy().ravel()
print(merge_df_data_test2_prepared.shape)

# COMMAND ----------

#visualize the data
pd.DataFrame(merge_df_data_test2_prepared).hist(bins=50, figsize=(20,15))
plt.show()

# COMMAND ----------

# model prediction
log_pred_test2 = log_clf.predict(merge_df_data_test2_prepared)
log_scores_test2 = log_clf.decision_function(merge_df_data_test2_prepared)
svc_pred_test2 = svc_clf.predict(merge_df_data_test2_prepared)
svc_scores_test2 = svc_clf.decision_function(merge_df_data_test2_prepared)
rnd_pred_test2 = rnd_clf.predict(merge_df_data_test2_prepared)
rnd_scores_test2 = rnd_clf.predict_proba(merge_df_data_test2_prepared)

# COMMAND ----------

# Logistic Regression Performance Measures
conf_matrx_test2 = confusion_matrix(merge_df_label_test2_prepared, log_pred_test2)
print(conf_matrx_test2)
log_acc_test2 = (conf_matrx_test2[0,0] + conf_matrx_test2[1,1]) / np.sum(conf_matrx_test2)
log_precision_test2 = precision_score(merge_df_label_test2_prepared, log_pred_test2)
log_recall_test2 = recall_score(merge_df_label_test2_prepared, log_pred_test2)
log_f1_test2 = f1_score(merge_df_label_test2_prepared, log_pred_test2)
print("Logistic Regression accuracy is: {}".format(np.round(log_acc_test2, 4)))
print("Logistic Regression precision is: {}".format(np.round(log_precision_test2, 4)))
print("Logistic Regression recall is: {}".format(np.round(log_recall_test2, 4)))
print("Logistic Regression F1 score is: {}".format(np.round(log_f1_test2, 4)))

# COMMAND ----------

# SVM Performance Measures
conf_matrx_test2 = confusion_matrix(merge_df_label_test2_prepared, svc_pred_test2)
print(conf_matrx_test2)
svc_acc_test2 = (conf_matrx_test2[0,0] + conf_matrx_test2[1,1]) / np.sum(conf_matrx_test2)
svc_precision_test2 = precision_score(merge_df_label_test2_prepared, svc_pred_test2)
svc_recall_test2 = recall_score(merge_df_label_test2_prepared, svc_pred_test2)
svc_f1_test2 = f1_score(merge_df_label_test2_prepared, svc_pred_test2)
print("SVM accuracy is: {}".format(np.round(svc_acc_test2, 4)))
print("SVM precision is: {}".format(np.round(svc_precision_test2, 4)))
print("SVM recall is: {}".format(np.round(svc_recall_test2, 4)))
print("SVM F1 score is: {}".format(np.round(svc_f1_test2, 4)))

# COMMAND ----------

# Random Forest Performance Measures
conf_matrx_test2 = confusion_matrix(merge_df_label_test2_prepared, rnd_pred_test2)
print(conf_matrx_test2)
rnd_acc_test2 = (conf_matrx_test2[0,0] + conf_matrx_test2[1,1]) / np.sum(conf_matrx_test2)
rnd_precision_test2 = precision_score(merge_df_label_test2_prepared, rnd_pred_test2)
rnd_recall_test2 = recall_score(merge_df_label_test2_prepared, rnd_pred_test2)
rnd_f1_test2 = f1_score(merge_df_label_test2_prepared, rnd_pred_test2)
print("Random Forest accuracy is: {}".format(np.round(rnd_acc_test2, 4)))
print("Random Forest precision is: {}".format(np.round(rnd_precision_test2, 4)))
print("Random Forest recall is: {}".format(np.round(rnd_recall_test2, 4)))
print("Random Forest F1 score is: {}".format(np.round(rnd_f1_test2, 4)))

# COMMAND ----------

keras_model=keras.models.load_model("/dbfs/mnt/bdh_mlai_mnt/yjin2/FSMsPie/dnn_model.h5")

dnn_acc_test2 = keras_model.evaluate(merge_df_data_test2_prepared, merge_df_label_test2_prepared)
dnn_scores_test2 = keras_model.predict(merge_df_data_test2_prepared).ravel()
print("%s: %.2f%%" % (keras_model.metrics_names[0], dnn_acc_test2[0]*100))
print("%s: %.2f%%" % (keras_model.metrics_names[1], dnn_acc_test2[1]*100))

dnn_bin_scores_test2 = np.round(dnn_scores_test2)
conf_matrx_test2 = confusion_matrix(merge_df_label_test2_prepared, dnn_bin_scores_test2)
print(conf_matrx_test2)
dnn_acc_test2 = (conf_matrx_test2[0,0] + conf_matrx_test2[1,1]) / np.sum(conf_matrx_test2)
dnn_precision_test2 = precision_score(merge_df_label_test2_prepared, dnn_bin_scores_test2)
dnn_recall_test2 = recall_score(merge_df_label_test2_prepared, dnn_bin_scores_test2)
dnn_f1_test2 = f1_score(merge_df_label_test2_prepared, dnn_bin_scores_test2)
print("Deep Neural Network accuracy is: {}".format(np.round(dnn_acc_test2, 4)))
print("Deep Neural Network precision is: {}".format(np.round(dnn_precision_test2, 4)))
print("Deep Neural Network recall is: {}".format(np.round(dnn_recall_test2, 4)))
print("Deep Neural Network F1 score is: {}".format(np.round(dnn_f1_test2, 4)))


auc_dnn_test2 = roc_auc_score(merge_df_label_test2_prepared, dnn_scores_test2)
print("DNN AUC is: {:.4f}".format(auc_dnn_test2))
