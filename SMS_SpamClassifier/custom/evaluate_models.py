# #!/usr/bin/env python
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, \
    ShuffleSplit, StratifiedKFold, learning_curve
from sklearn.metrics import make_scorer, accuracy_score, \
    recall_score, confusion_matrix

# Plot Learning Curves
def train_plot(clf, X, y, cv, verbose, train_sizes, n_jobs, 
               scorer_, metric, axes, axis):
    """
    Trains and plots learning_curves, given:
        scorer_: a make_scorer object
        metric: str, the name of the metric
    ...and all other args passed to learning_curve in the 
    plot_learning_curve function.
    """
    # train
    train_sizes, train_scores, test_scores = \
        learning_curve(clf, X, y, cv=cv, verbose=verbose,
                       train_sizes=train_sizes, n_jobs=n_jobs,
                       scoring=scorer_)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)    
    
    # plot
    axes[axis].grid()
    axes[axis].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[axis].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[axis].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[axis].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[axis].legend(loc="lower right")
    axes[axis].set_ylabel(metric)
    

def plot_learning_curve(clf, title, X, y, axes=None, ylim=(.95, 1.01), 
                        cv=5, train_sizes=np.linspace(.1, 1.0, 5),
                        verbose=0, n_jobs=-1):
    """
    Adapted from the Plot Learning Curves example in Scikit-Learn 
    to show only the performance of a scorer, using accuracy, 
    sensitivity, and specificity as metrics.
    """
    # set axes, title, ylims, xlabel
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 12))
    
    axes[0].set_title(title)
    axes[2].set_xlabel("Training Examples")
    
    if ylim is not None:
        axes[0].set_ylim(*ylim)
        axes[1].set_ylim(*ylim)
        axes[2].set_ylim(*ylim)

    # setup scorers
    scorers = {
        'acc': make_scorer(accuracy_score),
        'tpr': make_scorer(recall_score, pos_label=1), # sensitivity
        'tnr': make_scorer(recall_score, pos_label=0) # specificity
    }

    # plots
    train_plot(clf, X, y, cv=cv, verbose=verbose, 
               train_sizes=train_sizes, n_jobs=n_jobs, 
               scorer_=scorers['acc'], metric='Accuracy', 
               axes=axes, axis=0)
    train_plot(clf, X, y, cv=cv, verbose=verbose, 
               train_sizes=train_sizes, n_jobs=n_jobs, 
               scorer_=scorers['tpr'], metric='Sensitivity', 
               axes=axes, axis=1)
    train_plot(clf, X, y, cv=cv, verbose=verbose, 
               train_sizes=train_sizes, n_jobs=n_jobs, 
               scorer_=scorers['tnr'], metric='Specificity', 
               axes=axes, axis=2)
    
    return plt


def compare_two_classifiers(X, y, clf1, clf2, title1, title2, cv):
    """
    Wrapper for plot_learning_curve with a timer.
    Compares two classifiers' learning curves.
    """
    t = time.time()
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    plot_learning_curve(clf1, title1, X, y, axes=axes[:, 0], cv=cv)
    plot_learning_curve(clf2, title2, X, y, axes=axes[:, 1], cv=cv)
    m, s = divmod(time.time() - t, 60)
    print(f'Elapsed: {m:0.0f} m {s:0.0f} s')
    plt.show()   