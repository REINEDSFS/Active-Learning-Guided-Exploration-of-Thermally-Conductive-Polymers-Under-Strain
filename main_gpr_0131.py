# import the package you need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor, kernels as K

from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
polymer_embedding_model = word2vec.Word2Vec.load('POLYINFO_PI1M.pkl')



with open('deformed1.csv') as csvfile:
  reader = csv.reader(csvfile)
  rows = [row for row in reader]
data = np.array(rows)
data = np.delete(data, 0, axis=0)
sentences = list()
smiles = list(data[:,0])

# the following 4 lines are important to understand how the encoding process works (I can also not explain them explicitly to you :D)
for i in range(len(smiles)):
    sentence = MolSentence(mol2alt_sentence(Chem.MolFromSmiles(smiles[i]), 1))
    sentences.append(sentence)

polymer_embeddings = [DfVec(x) for x in sentences2vec(sentences, polymer_embedding_model, unseen='UNK')]
X = np.array([x.vec.tolist() for x in polymer_embeddings]) #X as feature matrix
y = np.array(data[:,1]).astype(np.float64)  #y as lable

# use 5-fold cross validation and grid search
# to find the best hyperparameters of GPR
# kernels = [
#     K.RBF() + K.WhiteKernel(),
#     K.Matern() + K.WhiteKernel(),
#     K.RationalQuadratic() + K.WhiteKernel(),
#     K.ExpSineSquared() + K.WhiteKernel(),
#     K.DotProduct() + K.WhiteKernel(),
#     K.RBF() + K.ConstantKernel(),
# ]

# params = {
#     'kernel': kernels,
#     'alpha': [1e-12, 1e-8, 1e-4, 1],

# }

# gpr = GaussianProcessRegressor(random_state=42)
# clf = GridSearchCV(gpr, params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
# clf.fit(X, y)
# print(clf.best_params_, clf.best_score_)


MSEs = [] # mean square error
R2s = [] # coefficient of determination
gpr = GaussianProcessRegressor(
    kernel= K.RationalQuadratic() + K.WhiteKernel(),
    alpha=1e-4,
    random_state=42,
)
# gpr = clf.best_estimator_
kf = KFold(n_splits=5)

y_true_plot=[]
y_pred_plot=[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    gpr.fit(X_train, y_train) #use the training data to fit the model
    y_pred = gpr.predict(X_test) # use the trained model to make prediction
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    R2s.append(r2)
    MSEs.append(mse)
    y_true_plot.extend(y_test)
    y_pred_plot.extend(gpr.predict(X_test))
print(np.mean(R2s))
print(np.std(R2s))
print(np.mean(MSEs))
print(np.std(MSEs))

max_xy = max(max(y_true_plot), max(y_pred_plot))
min_xy = min(min(y_true_plot), min(y_pred_plot))

fig, ax = plt.subplots()
plt.scatter(y_true_plot, y_pred_plot,s=1)
plt.plot([min_xy, max_xy], [min_xy, max_xy], 'r')
bbox = dict(boxstyle="round", fc='1',alpha=0.5)
plt.annotate('$R^2=%.2f$' % (np.mean(R2s)), (0.05, 0.9), size=10, bbox=bbox, xycoords='axes fraction')
plt.xlabel("True value of TC(C)")
plt.ylabel("Predicted value of TC(C)")
plt.axis('square')
plt.xlim(min_xy, max_xy)
plt.ylim(min_xy, max_xy)
plt.show()

y_mean, y_std = gpr.predict(X, return_std=True)
print(y_std)


df = pd.read_csv('./dataset2.csv')

# the following 4 lines are important to understand how the encoding process works (I can also not explain them explicitly to you :D)
sentences = []
smiles = df['SMILES'].values
for i in range(len(smiles)):
    sentence = MolSentence(mol2alt_sentence(Chem.MolFromSmiles(smiles[i]), 1))
    sentences.append(sentence)

polymer_embeddings = [DfVec(x) for x in sentences2vec(sentences, polymer_embedding_model, unseen='UNK')]
X = np.array([x.vec.tolist() for x in polymer_embeddings]) #X as feature matrix
y_mean, y_std = gpr.predict(X, return_std=True)

df['TC_pred'] = y_mean
df['TC_std'] = y_std
df.to_csv('dataset2_pred.csv', index=False)

plt.figure(figsize=(10, 5))
# idx = np.argsort(y_mean)
# y_mean = y_mean[idx]
# y_std = y_std[idx]
plt.plot(np.arange(len(y_mean)), y_mean, '.', label='Prediction', ms=2)
plt.fill_between(np.arange(len(y_mean)), y_mean - y_std, y_mean + y_std, alpha=0.5, label='Uncertainty')
#plt.plot(np.arange(len(y_mean)), y, label='True value')
plt.legend()
plt.show()
