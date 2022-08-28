import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab

log = logging.getLogger(__name__)

def plot_accuracies(epochs, accuracies, labels, title):

  fig_ = plt.figure(figsize=(32, 16))

  for i in range(len(accuracies)):

    plt.plot(epochs, accuracies[i], label=labels[i])

  plt.ylim(0.0, 1.0)
  plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
  plt.grid()

  plt.legend(loc=2, ncol=1)
  plt.title(title)
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy (%)")
  plt.show()