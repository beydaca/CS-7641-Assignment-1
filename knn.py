'''knn.py

K-nearest neighbors
'''
import logging
import matplotlib.pyplot as plt
import numpy as np
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
logger = logging.getLogger(__name__)

class KNN(Experiment):

  def __init__(self, attributes, classifications, dataset, **kwargs):
      ''' Construct the object
      '''
      pipeline = Pipeline([('scale', StandardScaler()), ('predict', KNeighborsClassifier())])
      params = {
        'predict__metric':['manhattan','euclidean'],
        'predict__n_neighbors': np.arange(1, 30, 3),
      }
      learning_curve_train_sizes = np.arange(0.1, 1.0, 0.025)
      super().__init__(attributes, classifications, dataset, 'knn', pipeline, params, 
        learning_curve_train_sizes, True)

  # def run(self):
  #
  #     cv_best = super().run()
  #     x_train, x_test, y_train, y_test = self._split_train_test()
  #     clf = cv_best.best_estimator_
  #     metrics = ['manhattan','euclidean']
  #     neighbors = np.arange(1, 30, 3)
  #     for metric in metrics:
  #         scores = []
  #         for neighbor in neighbors:
  #             clf.set_params(**{'predict__metric': metric, 'predict__n_neighbors': neighbor})
  #             clf.fit(x_train, y_train)
  #             score = clf.score(x_test, y_test)
  #             scores.append(score)
  #         csv_str = '{}/{}'.format(self._dataset, self._algorithm)
  #         plt.clf()
  #         plt.figure(4)
  #         plt.plot(neighbors, scores, marker='o', color='blue')
  #         plt.grid(linestyle='dotted')
  #         plt.xlabel('K neighbors')
  #         plt.ylabel('Score')
  #         plt.savefig('./results/{}/knn_{}.png'.format(csv_str, metric))

