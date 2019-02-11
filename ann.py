'''ann.py

Artifical Neural Networks
'''
import logging
import matplotlib.pyplot as plt
import numpy as np
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
logger = logging.getLogger(__name__)

class ANN(Experiment):

  def __init__(self, attributes, classifications, dataset, **kwargs):
      ''' Construct the object
      '''
      pipeline = Pipeline([('scale', StandardScaler()), ('predict', MLPClassifier(random_state=10, max_iter=2000, early_stopping=True))])
      params = {
        'predict__activation':['logistic', 'relu'],
        'predict__hidden_layer_sizes': [(32), (64), (128), (32, 64, 32), (64, 128, 64)]
      }
      learning_curve_train_sizes = np.arange(0.05, 1.0, 0.025)
      super().__init__(attributes, classifications, dataset, 'ann', pipeline, params, 
        learning_curve_train_sizes, True, verbose=1, iteration_curve=True)

  # def run(self):
  #     '''Run the expierment, but we need to check the prune size
  #     '''
  #
  #     cv_best = super().run()
  #     x_train, x_test, y_train, y_test = self._split_train_test()
  #     clf = cv_best.best_estimator_
  #     functions = ['logistic', 'relu']
  #     hidden_layers = [(32), (64), (128)]
  #     for function in functions:
  #         scores = []
  #         for hidden_layer in hidden_layers:
  #             clf.set_params(**{'predict__activation': function, 'predict__hidden_layer_sizes': hidden_layer})
  #             clf.fit(x_train, y_train)
  #             score = clf.score(x_test, y_test)
  #             scores.append(score)
  #         csv_str = '{}/{}'.format(self._dataset, self._algorithm)
  #         plt.clf()
  #         plt.figure(4)
  #         plt.plot(hidden_layers, scores, marker='o', color='blue')
  #         plt.grid(linestyle='dotted')
  #         plt.xlabel('Hidden Layer size')
  #         plt.ylabel('Score')
  #         plt.savefig('./results/{}/ann_{}.png'.format(csv_str, function))
