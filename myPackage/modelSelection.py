import matplotlib.pyplot as plt
from sklearn import model_selection

def modelSelection(data, labels, models, seed, score, plot= False):
    results= []
    names = []
    model_results = {}

    for name, model in models.items():
        kfold = model_selection.KFold(n_splits= 10, random_state= seed)
        cv_results = model_selection.cross_val_score(model, data, labels, cv= kfold, scoring= score)
        results.append(cv_results)
        names.append(name)
        model_results[name] = cv_results.mean()
        print("Estimator '{}': {:0.4f} for '{}' (+/-{:0.04f})".format(name, cv_results.mean(), score, cv_results.std()))

    # boxplot algorithm comparison
    if plot:
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        ax.set_ylabel('Accuracy')
        plt.show()

    model_results_sorted = sorted(model_results, key=model_results.__getitem__, reverse=True)
    print("\nModels ordered by accuracy: \n{}".format(model_results_sorted))

    return [model_results_sorted[0], model_results[model_results_sorted[0]]]
