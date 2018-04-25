import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import csv
from sklearn.utils import Bunch
import pickle
# #############################################################################
def load_data(data_file_name):

    with open(data_file_name) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[1:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)
    return Bunch(data=data, target=target,
          target_names=target_names,
          DESCR=None,
          feature_names=np.asarray(['Variance-pixels’ projection', 'Gravity-X',
                         'Gravity-Y', 'aspect_ratio','extent','RPA','Convecity','Rectangularity']))
# Load data

data=load_data('./dataset/data.csv')
X, y = shuffle(data.data, data.target, random_state=13)

X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# #############################################################################
# Fit regression model
params = {'n_estimators': 600, 'max_depth': 4, 'min_samples_split': 3,
          'learning_rate': 0.005, 'loss': 'huber'}
#对于回归模型，有均方差"ls", 绝对损失"lad", Huber损失"huber"和分位数损失“quantile”。
#一般来说，如果数据的噪音点不多，用默认的均方差"ls"比较好。如果是噪音点较多，则推荐用抗噪音的损失函数"huber"。而如果我们需要对训练集进行分段预测的时候，则采用“quantile”
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))/X_test.shape[0]

print("AVG MSE: %.4f" % mse)
filename = 'finalized_model.sav'
# pickle.dump(clf, open(filename, 'wb'))

# #############################################################################
# Plot training deviance
# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# #############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance)# / (feature_importance.max()))
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, data.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Feature Importance')
plt.show()
