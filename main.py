import sys

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import random as r
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score


def get_data():
    '''

    :return: tuple of data_train, data_test, dict_mapping
    '''
    df = pd.read_csv('abalone.data',
                     names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight',
                            'Shell weight', 'Rings'], header=None)

    le = preprocessing.LabelEncoder()
    dict = {}
    y_labels = ['infant', 'child', 'adult', 'mid_age', 'old', 'very_old']
    unique = df['Sex'].unique()
    df['Sex'] = le.fit_transform(df['Sex'])
    trans = le.transform(unique)
    dict['Sex'] = [(unique[i], trans[i]) for i in range(len(unique))]
    y = df['Rings'] + 1.5
    y = pd.cut(y, 6, labels=y_labels)
    df.drop(['Rings'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    ax1.set_title('training set')
    ax2.set_title('testing set')
    y_train.value_counts().plot(kind='bar', ax=ax1)
    y_test.value_counts().plot(kind='bar', ax=ax2)
    plt.savefig('training_testing.png')

    return X_train, X_test, y_train, y_test, y_labels, dict


def plot_graph(x_train, x_label, y_train, y_label, x_test, y_test, title):
    plt.clf()
    plt.plot(x_train, y_train, label='training')
    plt.plot(x_test, y_test, label='testing')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(title)


def decision_tree(df_train, y_train, df_test, y_test, labels):
    x_train = []
    y_train_ = []
    x_test = []
    y_test_ = []
    for i in range(3, 10, 1):
        clf = DecisionTreeClassifier(max_depth=i)
        clf.fit(df_train, y_train)
        y_t_hat = clf.predict(df_train)
        score_training = accuracy_score(y_t_hat, y_train)
        y_train_.append(score_training)
        x_train.append(i)
        y_p = clf.predict(df_test)
        score_testing = accuracy_score(y_test, y_p)
        y_test_.append(score_testing)
        x_test.append(i)
        print_results(score_training, score_testing, y_test, y_p, f'Decision Tree - {i}')
    plot_graph(x_train, 'Max Depth', y_train_, 'Accuracy', x_test, y_test_, 'dt_max-depth_com.png')
    x_train = []
    y_train_ = []
    x_test = []
    y_test_ = []
    for i in range(3, 10, 1):
        clf = DecisionTreeClassifier(max_leaf_nodes=i)
        clf.fit(df_train, y_train)
        y_t_hat = clf.predict(df_train)
        score_training = accuracy_score(y_t_hat, y_train)
        y_train_.append(score_training)
        x_train.append(i)
        y_p = clf.predict(df_test)
        score_testing = accuracy_score(y_test, y_p)
        y_test_.append(score_testing)
        x_test.append(i)
        print_results(score_training, score_testing, y_test, y_p, f'Decision Tree leaf- {i}')
    plot_graph(x_train, 'Max Leaf Node', y_train_, 'Accuracy', x_test, y_test_, 'dt_max_leaf_node.png')
    title = r"Decision Tree Learning Curves"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = DecisionTreeClassifier(max_leaf_nodes=8)
    plt.clf()
    _plt = plot_learning_curve(estimator, title, df_train, y_train, axes=None, ylim=None, cv=cv, n_jobs=4)
    _plt.savefig('dt_learning_curve.png')
    estimator.fit(df_train, y_train)
    X_test = estimator.predict(df_test)
    graph_confusion_matrix(y_test, X_test, labels, 'dt_confusion_matrix.png')


def graph_confusion_matrix(y_test, pred, labels, file_name):
    unique_elements, counts_elements = np.unique(pred, return_counts=True)
    print(unique_elements, counts_elements)
    plt.clf()
    cm = confusion_matrix(y_test, pred, labels)
    print(cm)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt=".5g", square=True)
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    b, t = ax.get_ylim()
    ax.set_ylim(b + 0.5, t - 0.5)
    plt.savefig(file_name)


def neural_network(X_train, y_train, X_test, y_test, labels):
    y_tr = []
    y_ts = []
    x_label = []
    for i in range(2, 5, 1):
        for j in range(2, 8, 1):
            clf = MLPClassifier(early_stopping=True, hidden_layer_sizes=(5, 2))
            clf.fit(X_train, y_train)
            y_t_hat = clf.predict(X_train)
            score_training = accuracy_score(y_t_hat, y_train)
            y_tr.append(score_training)
            y_p = clf.predict(X_test)
            score_testing = accuracy_score(y_test, y_p)
            y_ts.append(score_testing)
            x_label.append(f'{j}*{i}')
            print_results(score_training, score_testing, y_test, y_p, f'Neural Network ({j}*{i})')
    plot_graph(x_label, 'Hidden Layer Size (no of Neurons * no of Layer)', y_tr, 'Accuracy', x_label, y_ts,
               'nn_hidder_layers.png')
    title = r"Neutral Network Learning Curves"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = MLPClassifier(early_stopping=True, hidden_layer_sizes=(6, 4))
    plt.clf()
    _plt = plot_learning_curve(estimator, title, X_train, y_train, axes=None, ylim=None, cv=cv, n_jobs=4)
    _plt.savefig('nn_learning_curve.png')
    estimator.fit(X_train, y_train)
    X_test = estimator.predict(X_test)
    graph_confusion_matrix(y_test, X_test, labels, 'nn_confusion_matrix.png')


def boosted_decision_tree(X_train, y_train, X_test, y_test, labels):
    x_train = []
    y_train_ = []
    x_test = []
    y_test_ = []
    for i in range(3, 10, 1):
        clf = GradientBoostingClassifier(max_depth=i)
        clf.fit(X_train, y_train)
        y_t_hat = clf.predict(X_train)
        score_training = accuracy_score(y_t_hat, y_train)
        y_train_.append(score_training)
        x_train.append(i)
        y_p = clf.predict(X_test)
        score_testing = accuracy_score(y_test, y_p)
        y_test_.append(score_testing)
        x_test.append(i)
        print_results(score_training, score_testing, y_test, y_p, f'Gradient Boosting Decision Tree - max_depth= {i}')
    plot_graph(x_train, 'Max Depth', y_train_, 'Accuracy', x_test, y_test_, 'boosting_dt_max-depth_com.png')
    x_train = []
    y_train_ = []
    x_test = []
    y_test_ = []
    for i in range(3, 10, 1):
        clf = GradientBoostingClassifier(max_leaf_nodes=i)
        clf.fit(X_train, y_train)
        y_t_hat = clf.predict(X_train)
        score_training = accuracy_score(y_t_hat, y_train)
        y_train_.append(score_training)
        x_train.append(i)
        y_p = clf.predict(X_test)
        score_testing = accuracy_score(y_test, y_p)
        y_test_.append(score_testing)
        x_test.append(i)
        print_results(score_training, score_testing, y_test, y_p,
                      f'Gradient Boosting Decision Tree - max_leaf_nodes= {i}')
    plot_graph(x_train, 'Max Leaf Node', y_train_, 'Accuracy', x_test, y_test_, 'boosting_dt_max_leaf_node.png')
    title = r"Gradient Boosting Decision Tree Curves"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = GradientBoostingClassifier(max_leaf_nodes=8)
    plt.clf()
    _plt = plot_learning_curve(estimator, title, X_train, y_train, axes=None, ylim=None, cv=cv, n_jobs=4)
    _plt.savefig('boosting_dt_learning_curve.png')
    estimator.fit(X_train, y_train)
    prod = estimator.predict(X_test)
    graph_confusion_matrix(y_test, prod, labels, 'boosting_dt_confusion_matrix.png')


def support_vector_machine(X_train, y_train, X_test, y_test, labels):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    x_labels = []
    test_scores = []
    training_scores = []
    for k in kernels:
        clf = SVC(kernel=k)
        x_labels.append(k)
        clf.fit(X_train, y_train)
        y_t_hat = clf.predict(X_train)
        score_training = accuracy_score(y_t_hat, y_train)
        training_scores.append(score_training)
        y_p = clf.predict(X_test)
        score_testing = accuracy_score(y_test, y_p)
        test_scores.append(score_testing)
        print_results(score_training, score_testing, y_test, y_p, f'Support Vector Machine - kernel= {k}')
    plot_graph(x_labels, 'Kernels', training_scores, 'Accuracy', x_labels, test_scores, 'svc_kernels.png')
    title = r"Support Vector Machines Learning Curves"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = SVC(kernel='rbf')
    plt.clf()
    _plt = plot_learning_curve(estimator, title, X_train, y_train, axes=None, ylim=None, cv=cv, n_jobs=4)
    _plt.savefig('svc_learning_curve.png')
    estimator.fit(X_train, y_train)
    prod = estimator.predict(X_test)
    graph_confusion_matrix(y_test, prod, labels, 'svc_confusion_matrix.png')


def print_results(score_training, score_testing, y_test, predicted, title):
    precision, recall, fscore, support = score(y_test, predicted)
    print(f'Result: {title}')
    print(f'Score training: {score_training}')
    print(f'Score testing: {score_testing}')
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    print('------------------------------------------------------')


def k_nearest_neighbors(X_train, y_train, X_test, y_test, labels):
    x_labels = []
    train_scores = []
    test_scores = []
    for i in range(2, 8, 1):
        clf = KNeighborsClassifier(n_neighbors=i)
        x_labels.append(i)
        clf.fit(X_train, y_train)
        y_t_hat = clf.predict(X_train)
        score_training = accuracy_score(y_t_hat, y_train)
        train_scores.append(score_training)
        y_p = clf.predict(X_test)
        score_testing = accuracy_score(y_test, y_p)
        test_scores.append(score_testing)
        print_results(score_training, score_testing, y_test, y_p, f'K Nearest Neighbors, neighbors= {i}')
    plot_graph(x_labels, 'No. Neighbors ', train_scores, 'Accuracy', x_labels, test_scores, 'kn_neighbors.png')
    title = r"K-Nearest Neighbors"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = KNeighborsClassifier(n_neighbors=5)
    plt.clf()
    _plt = plot_learning_curve(estimator, title, X_train, y_train, axes=None, ylim=None, cv=cv, n_jobs=4)
    _plt.savefig('kn_neighbors_learning_curve.png')
    estimator.fit(X_train, y_train)
    prod = estimator.predict(X_test)
    graph_confusion_matrix(y_test, prod, labels, 'kn_neighbors_confusion_matrix.png')


'''   COPY FROM SKLEARN WEBSITE'''


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Accuracy")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Performance of the model")

    return plt


def run_experiment(X_train, y_train, X_test, y_test, labels):
    exp = [(DecisionTreeClassifier(max_leaf_nodes=8), 'Decision Tree'),
           (MLPClassifier(early_stopping=True, hidden_layer_sizes=(6, 4)), 'Neural Network'),
           (GradientBoostingClassifier(max_leaf_nodes=8), 'Gradient Boosting Decision Tree'),
           (SVC(kernel='rbf'), 'Support Vector Machine'), (KNeighborsClassifier(n_neighbors=5), 'K Nearest Neighbors')]
    for e in exp:
        clf = e[0]
        print(clf.get_params())
        clf.fit(X_train, y_train)
        y_t_hat = clf.predict(X_train)
        score_training = accuracy_score(y_t_hat, y_train)
        y_p = clf.predict(X_test)
        score_testing = accuracy_score(y_test, y_p)
        print_results(score_training, score_testing, y_test, y_p, e[1])


def main_part():
    r.seed(526234)
    np.random.seed(526234)
    X_train, X_test, y_train, y_test, labels, _ = get_data()
    print(f'training {X_train.shape}, testing {X_test.shape}')
    print('train set count')
    print(y_train.value_counts())
    print('test set class count')
    print(y_test.value_counts())
    args = sys.argv[1:]
    method = args[0] if len(args) > 0 else ''
    if method == 'dt':
        decision_tree(X_train, y_train, X_test, y_test, labels)
    elif method == 'nn':
        neural_network(X_train, y_train, X_test, y_test, labels)
    elif method == 'bdt':
        boosted_decision_tree(X_train, y_train, X_test, y_test, labels)
    elif method == 'svm':
        support_vector_machine(X_train, y_train, X_test, y_test, labels)
    elif method == 'knn':
        k_nearest_neighbors(X_train, y_train, X_test, y_test, labels)
    else:
        run_experiment(X_train, y_train, X_test, y_test, labels)


if __name__ == "__main__":
    main_part()
