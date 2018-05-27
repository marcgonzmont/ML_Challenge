from myPackage import modelSelection as ms
from myPackage import ComponentSelection as CS
from myPackage import tools as tl
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from pandas import read_csv
import argparse
import time
import pickle

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data",
                    help="-d Training data")
    ap.add_argument("-r", "--results", required= False,
                    help="-r Results path")
    args = vars(ap.parse_args())

    # Create results folder
    if args.get("results") is not None:
        if not exists(args["results"]):
            tl.makeDir(results_folder)
        results_folder = args["results"]
    else:
        results_folder = "."

    # Configuration
    pca_threshold = 1.5
    val_ratio = 0.2
    plot = True
    max_estimators = 201
    ini_estimators = 10
    step_estimators = 10
    max_depth = 11
    seed = 10
    models = {}
    loop = 5
    scores = ['precision', 'recall', 'f1']
    class_names = ['trees', 'grass', 'soil', 'concrete', 'asphalt', 'buildings', 'cars', 'pools', 'shadows']

    # models['DTC'] = DecisionTreeClassifier(max_depth=None)
    # models['RFC'] = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    # models['GNB'] =  GaussianNB()
    # models['BGG-DCT'] = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=None), n_estimators=500, n_jobs=-1)
    # models['BGG-RFC'] = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=500, n_jobs=-1), n_estimators=50, n_jobs=-1)
    # models['BGG-GNB'] = BaggingClassifier(base_estimator=GaussianNB(), n_estimators=500, n_jobs=-1)


    models['DTC'] = DecisionTreeClassifier()
    models['RFC'] = RandomForestClassifier()
    models['GNB'] = GaussianNB()
    models['BGG-DCT'] = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10,
                                          n_jobs=-1)
    models['BGG-RFC'] = BaggingClassifier(base_estimator=RandomForestClassifier(n_jobs=-1),
                                          n_estimators=10, n_jobs=-1)
    models['BGG-GNB'] = BaggingClassifier(base_estimator=GaussianNB(), n_estimators=10, n_jobs=-1)

    score = 'accuracy'
    # Training and validating data
    all_data = read_csv(args["data"], header= 0).values
    tr_data, val_data = tl.split_train_test(all_data, val_ratio)
    data, labels = tr_data[:,1:], tr_data[:, 0]
    data = tl.normalize(data)
    val_data, val_labels = val_data[:,1:], val_data[:, 0]
    val_data = tl.normalize(val_data)

    # Apply PCA
    # n_components = CS.varianceStudio(data.copy(), pca_threshold)
    # new_data = CS.componentSelection(data.copy(), n_components)
    # new_val_data = CS.componentSelection(val_data.copy(), n_components)
    #
    #
    # print("File contains {} elements\n"
    #       "Training data contains {} features --> PCA {} features\n"
    #       "Validating data contains {} features --> PCA {} features\n\n".format(len(all_data), len(tr_data[0]), len(new_data[0]), len(val_data), len(new_val_data[0])))

    # Loop to look for the best model in some epochs
    names_acc = []
    for i in range(loop):
        print("\nLoop {}/{}".format(i+1, loop))
        model_name = ms.modelSelection(data, labels, models, seed, score, plot)
        names_acc.append(model_name)
    # print("\nWinners:\n{}".format(names_acc))

    # Sort the names of the models by accuracy
    names_acc_sorted = sorted(names_acc, key= lambda tup: tup[1], reverse= True)
    print("\nWinners sorted by accuracy:\n{}".format(names_acc_sorted))

    names = [item[0] for item in names_acc_sorted]

    count={}
    for model in models.keys():
        count[model] = names.count(model)
    models_sorted = sorted(count, key=count.__getitem__, reverse=True)
    model_count = models_sorted[0]
    model_acc = names[0]
    # print("Best model by count:\n{}\n"
    #       "Best model by accuracy:\n{}\n\n".format(model_count, model_acc))

    best_by_count = names_acc_sorted[names.index(model_count)]

    if model_count == model_acc:
        # print("\nEQUALS!!!")
        model_selected = names_acc_sorted[0]
    else:
        # print("\nDIFFERENTS!!!")
        if count[model_count] >= int(len(names)*0.6):
            model_selected = best_by_count
        else:
            model_selected = names_acc_sorted[0]

    print("\nBest model by accuracy:\n{}\n"
          "Best model by count:\n{}\n"
          "Model selected:\n{}\n\n".format(names_acc_sorted[0], best_by_count, model_selected))


    # models_conf = {}
    if model_selected[0] == 'DCT':
        depth = np.arange(4, max_depth, 1)
        # depth = np.append(depth, None)
        tuned_parameters = [{'max_depth': depth, 'presort': [False, True]}]

    elif model_selected[0] == 'RFC':
        estimators = np.arange(ini_estimators, max_estimators, step_estimators)
        tuned_parameters = [{'n_estimators': estimators}]

    elif model_selected[0] == 'GNB':
        print("hhh")
    # model_selected == 'BGG-RFC'
    elif model_selected[0] == 'BGG-DCT' or model_selected[0] == 'BGG-RFC':
        print("BGG-DCT, BGG-RFC")
        # depth = np.arange(5, max_depth, 5)
        estimators = np.arange(ini_estimators, max_estimators, step_estimators)
        # tuned_parameters_DCT = [{'max_depth': depth, 'presort': [False, True]}]
        tuned_parameters = [{'n_estimators': estimators}]

    elif model_selected[0] == 'BGG-GNB':
        estimators = np.arange(ini_estimators, max_estimators, step_estimators)
        tuned_parameters = [{'n_estimators': estimators}]

    # results = []
    best_estimator = None
    start = time.time()
    for score in scores:
        print("\nTuning hyper-parameters for '{}' model and '{}' score".format(model_selected[0], score))
        clf = GridSearchCV(models[model_selected[0]], tuned_parameters, cv= 10, scoring='%s_macro' % score, n_jobs= -1, verbose= 1)
        # clf.fit(new_data, labels)
        clf.fit(data, labels)
        print("\nBest parameters set found on development set:")
        print(clf.best_params_)
        print("\nBest score: {:0.3f}".format(clf.best_score_))
        print("\nGrid scores on development set for {}:".format(score))
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("{:0.3f} (+/-{:0.03f}) for {}".format(mean, std * 2, params))

        labels_pred = clf.best_estimator_.predict(val_data)
        print(classification_report(val_labels, labels_pred))
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(val_labels, labels_pred)
        tl.np.set_printoptions(precision=3)
        # Plot normalized confusion matrix
        tl.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                 title="Normalized confusion matrix for '{}' with '{}' estimator".format(score, model_selected[0]))
        modelName = tl.join(results_folder,"".join(("Marcos_GM_",model_selected[0],"_",str(score),".model")))
        pickle.dump(clf.best_estimator_, open(modelName, 'wb'))

    print("\n\nFINISHED!!!")
    print("\nExecution takes {:.4f} seconds for {} scores".format(time.time() - start, len(scores)))
