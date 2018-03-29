# from myPackage import ComponentSelection as CS
from myPackage import tools as tl
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import model_selection
from pandas import read_csv
import argparse
import pickle

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data",
                    help="-d Training data")
    ap.add_argument("-r", "--results",
                    help="-r Results path")
    args = vars(ap.parse_args())

    # Create results folder
    tl.makeDir(args["results"])

    # Configuration
    pca_threshold = 1.5
    seed = 10
    models = []
    models.append(('DTC', DecisionTreeClassifier(max_depth= None)))
    models.append(('RFC', RandomForestClassifier(n_estimators= 50, n_jobs= -1)))
    models.append(('GNB', GaussianNB()))
    models.append(('BGG-DCT', BaggingClassifier(base_estimator= DecisionTreeClassifier(max_depth= None), n_estimators= 50, n_jobs= -1)))
    models.append(('BGG-RFC', BaggingClassifier(base_estimator= RandomForestClassifier(n_estimators= 50, n_jobs= -1), n_estimators=50, n_jobs=-1)))
    models.append(('BGG-GNB', BaggingClassifier(base_estimator= GaussianNB(), n_estimators= 50, n_jobs= -1)))
    results= []
    names = []

    # depth = range(5, 101, 5)
    # tuned_parameterds_DTC = [{'max_depth': depth, 'presort': [False, True]}]
    # scores = ['precision']
    score = 'accuracy'

    tr_data = read_csv(args["data"], header= 0).values
    data, labels = tr_data[:,1:], tr_data[:, 0]

    # n_components = CS.varianceStudio(data.copy(), pca_threshold)
    # new_data = CS.componentSelection(data.copy(), n_components)

    # for name, model in models:
    # for score in scores:
    #     print("\nTuning hyper-parameters for 'DTC' model and '{}' score".format(score))
    #     clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameterds_DTC, cv= 10, scoring='%s_macro' % score)
    #     clf.fit(new_data, labels)
    #     results.append(clf.cv_results_)
    #     print("\nBest parameters set found on development set:")
    #     print(clf.best_params_)
    #     print("Best score: {:0.3f}".format(clf.best_score_))
        # print("\nGrid scores on development set:")
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("{:0.3f} (+/-{:0.03f}) for {}".format(mean, std * 2, params))
        # modelName = tl.altsep.join((args["results"],"".join(("Marcos_GM_DTC_",str(clf.best_params_['max_depth']),"_",str(score),".model"))))
        # pickle.dump(clf.best_estimator_, open(modelName, 'wb'))

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, data, labels, cv=kfold, scoring=score)
        results.append(cv_results)
        names.append(name)
        model_name = tl.altsep.join((args["results"],"".join(("Marcos_GM_", name))))
        print("Estimator '{}': {:0.3f} for '{}' (+/-{:0.03f})".format(name, cv_results.mean(), score, cv_results.std()))

    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    ax.set_ylabel('Accuracy')
    plt.show()