'''svm.py

Support Vector Machines
'''
import logging
import matplotlib.pyplot as plt
import numpy as np
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
logger = logging.getLogger(__name__)

class SVM(Experiment):

  def __init__(self, attributes, classifications, dataset, **kwargs):
      ''' Construct the object
      '''
      pipeline = Pipeline([('scale', StandardScaler()), ('predict', SVC())])
      params = {
        'predict__kernel': ['linear', 'poly', 'rbf'],
        # 'predict__C': 10.0 ** np.arange(-3, 8),
        # penalize distance, low = use all, high = use close b/c distance to decision boundry to penalized
        # 'predict__gamma': 10. ** np.arange(-5, 4),
        # 'predict__cache_size': [200],
        # 'predict__max_iter': [2000],
      }
      learning_curve_train_sizes = np.arange(0.05, 1.0, 0.05)
      super().__init__(attributes, classifications, dataset, 'svm', pipeline, params, 
        learning_curve_train_sizes, True, verbose=0, iteration_curve=True)

  # def run(self):
  #
  #     cv_best = super().run()
  #     x_train, x_test, y_train, y_test = self._split_train_test(.5)
  #     clf = cv_best.best_estimator_
  #     kernels = ['linear', 'poly', 'rbf']
  #     for kernel in kernels:
  #         scores = []
  #         clf.set_params(**{'predict__kernel': kernel})
  #         clf.fit(x_train, y_train)
  #         score = clf.score(x_test, y_test)
  #         scores.append(score)
  #         csv_str = '{}/{}'.format(self._dataset, self._algorithm)
  #         plt.clf()
  #         plt.figure(4)
  #         plt.bar(kernels, scores, 1/15, color='blue')
  #         plt.xlabel('Kernel')
  #         plt.ylabel('Score')
  #         plt.savefig('./results/{}/svm_kernels.png'.format(csv_str))
