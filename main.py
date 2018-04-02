from myPackage import modelSelection as ms
from myPackage import tools as tl
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import numpy as np
from pandas import read_csv
import argparse
# import pickle

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data",
                    help="-d Training data")
    ap.add_argument("-r", "--results",
                    help="-r Results path")
    args = vars(ap.parse_args())

    # Create results folder
    results_folder = args["results"]
    tl.makeDir(results_folder)

    # Configuration
    pca_threshold = 1.5
    plot = True
    max_estimators = 101
    max_depth = 101
    seed = 10
    models = {}
    # models['DTC'] = DecisionTreeClassifier(max_depth=None)
    # models['RFC'] = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    # models['GNB'] =  GaussianNB()
    # models['BGG-DCT'] = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=None), n_estimators=50, n_jobs=-1)
    # models['BGG-RFC'] = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=50, n_jobs=-1), n_estimators=50, n_jobs=-1)
    # models['BGG-GNB'] = BaggingClassifier(base_estimator=GaussianNB(), n_estimators=50, n_jobs=-1)
    # tuned_parameterds_DTC = [{'max_depth': depth, 'presort': [False, True]}]
    # scores = ['precision']
    models['DTC'] = DecisionTreeClassifier()
    models['RFC'] = RandomForestClassifier()
    models['GNB'] = GaussianNB()
    models['BGG-DCT'] = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10,
                                          n_jobs=-1)
    models['BGG-RFC'] = BaggingClassifier(base_estimator=RandomForestClassifier(n_jobs=-1),
                                          n_estimators=10, n_jobs=-1)
    models['BGG-GNB'] = BaggingClassifier(base_estimator=GaussianNB(), n_estimators=10, n_jobs=-1)

    score = 'accuracy'

    tr_data = read_csv(args["data"], header= 0).values
    data, labels = tr_data[:,1:], tr_data[:, 0]
    name=[]
    for i in range(5):
        model_name = ms.modelSelection(data, labels, models, seed, score, plot)
        name.append(model_name)
    print(name)
    # models_conf = {}
    # if model_name == 'DCT':
    #     depth = np.arange(5, max_depth, 5)
    #     # depth = np.append(depth, None)
    #     tuned_parameters = [{'max_depth': depth, 'presort': [False, True]}]
    #
    # elif model_name == 'RFC':
    #     estimators = np.arange(5,max_estimators,5)
    #     tuned_parameters = [{'n_estimators': estimators}]
    # elif model_name == 'GNB':
    #     print("hhh")
    # elif model_name == 'BGG-DCT':
    #     depth = np.arange(5, max_depth, 5)
    #     estimators = np.arange(5, max_estimators, 5)
    #     tuned_parameters_DCT = [{'max_depth': depth, 'presort': [False, True]}]
    #     tuned_parameters_BGG_DCT = [{'n_estimators': estimators}]
    # elif model_name == 'BGG-GNB':
    #     estimators = np.arange(5, max_estimators, 5)
    #     tuned_parameters_BGG_DCT = [{'n_estimators': estimators}]

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

