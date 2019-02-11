'''dt.py

Decision Trees
'''
import matplotlib.pyplot as plt
import logging 
import numpy as np
import pandas as pd
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
logger = logging.getLogger(__name__)

class DT(Experiment):

  def __init__(self, attributes, classifications, dataset, **kwargs):
      ''' Construct the object
      '''
      pipeline = Pipeline([('scale', StandardScaler()), ('predict', PrunedDecisionTreeClassifier())])
      params = {
        'predict__criterion': ['entropy'],
        'predict__class_weight': ['balanced'],
        'predict__max_depth': [None, 100, 200, 500],
        'predict__min_samples_leaf': [1, 2, 3, 4, 5],
      }
      learning_curve_train_sizes = np.arange(0.01, 1.0, 0.025)
      super().__init__(attributes, classifications, dataset, 'dt', pipeline, params, 
        learning_curve_train_sizes, True, verbose=1)

  # def run(self):
  #     '''Run the expierment, but we need to check the prune size
  #     '''
  #     cv_best = super().run()
  #     x_train, x_test, y_train, y_test = self._split_train_test()
  #     clf = cv_best.best_estimator_
  #     scores = []
  #     min_samples_leafs = [1, 2, 3, 4, 5]
  #     for min_samples_leaf in min_samples_leafs:
  #         clf.set_params(**{'predict__min_samples_leaf': min_samples_leaf, 'predict__alpha': 0})
  #         clf.fit(x_train, y_train)
  #         score = clf.score(x_test, y_test)
  #         scores.append(score)
  #     csv_str = '{}/{}'.format(self._dataset, self._algorithm)
  #     plt.figure(4)
  #     plt.plot(min_samples_leafs, scores, marker='o', color='blue')
  #     plt.grid(linestyle='dotted')
  #     plt.xlabel('Minimum Leaf Samples')
  #     plt.ylabel('Score')
  #     plt.savefig('./results/{}/min_leaf_samples.png'.format(csv_str))
      
class PrunedDecisionTreeClassifier(DecisionTreeClassifier):        
    def __init__(self,
               criterion='gini',
               splitter='best',
               max_depth=None,
               min_samples_split=2,
               min_samples_leaf=1,
               min_weight_fraction_leaf=0.,
               max_features=None,
               random_state=None,
               max_leaf_nodes=None,
               min_impurity_split=1e-7,
               class_weight=None,
               presort=False,
               alpha = 0):
      super().__init__(
          criterion=criterion,
          splitter=splitter,
          max_depth=max_depth,
          min_samples_split=min_samples_split,
          min_samples_leaf=min_samples_leaf,
          min_weight_fraction_leaf=min_weight_fraction_leaf,
          max_features=max_features,
          max_leaf_nodes=max_leaf_nodes,
          class_weight=class_weight,
          random_state=random_state,
          min_impurity_split=min_impurity_split,
          presort=presort)
      self.alpha = alpha
            
    def num_nodes(self):
        return  (self.tree_.children_left >= 0).sum()


