import joblib
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

from utils import utils


def build_ada_boosting_models(estimators, data_path):
    """
    :param estimators: diccionario
    :param data_path: path
    :return:
    """
    data = utils.loadData(data_path)
    n_columns = data.shape[1]
    x_instances = data[:, :n_columns - 1]
    y_instances = data[:, -1]

    source_folder = './adaboosting/models_and_evaluation'
    name_evaluation_file = source_folder + '/evaluacion.txt'
    name_model_file = '.'
    folders = source_folder + '/models'
    utils.create_folder(folders)

    algorithms = ['LR', 'CART', 'NAIVE', 'SGD', 'SVC', 'EXTRA']

    metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_id = []
    performance = []
    for i in range(len(algorithms)):
        performance.append([])

    for i, estimator in enumerate(algorithms):

        model_id.append(estimator)
        name_model_file = folders + '/' + estimator + '-adaboosting.pickle'
        adaboosting = AdaBoostClassifier(estimator=estimators[estimator],
                                         algorithm="SAMME"
                                         )
        for metrica in metricas:
            scoring = cross_val_score(adaboosting, x_instances, y_instances, scoring=metrica, cv=5)
            performance[i].append(round(scoring.mean(), 2))

        model = adaboosting.fit(x_instances, y_instances)
        joblib.dump(model, r'' + name_model_file)

    print("Modelos AdaBoosting construidos")
    results = pd.DataFrame(performance, columns=metricas, index=model_id)

    with open(name_evaluation_file, 'w') as f:
        utils.create_txt_report(results, model_id, f)

    name_graph = source_folder + '/adaboosting_barplot.png'
    title = 'Evaluación del modelo Adaboosting'
    xlabel = 'Estimadores'
    ylabel = 'Valor de las métricas'
    utils.plot_results(results, name_graph, title, xlabel, ylabel)
