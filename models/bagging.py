import joblib
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from utils import utils


def build_bagging_models(estimators, data_path):
    """
    :param estimators: diccionario
    :param data_path: path
    :return:
    """
    data = utils.loadData(data_path)
    n_columns = data.shape[1]
    x_instances = data[:, :n_columns - 1]
    y_instances = data[:, -1]

    source_folder = './bagging/models_and_evaluation'
    name_evaluation_file = source_folder + '/evaluacion.txt'
    folders = source_folder + '/models'
    utils.create_folder(folders)

    dict_exactitud = {
        'LR': [],
        'CART': [],
        'NAIVE': [],
        'KNN': [],
        'SGD': [],
        'SVC': [],
        'MLP': [],
        'EXTRA': []
    }

    metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_id = []
    performance = []
    for i in range(len(estimators.keys())):
        performance.append([])

    for i, estimator in enumerate(estimators.keys()):
        name_model_file = folders + '/' + estimator + '-bagging.pickle'
        bagging = BaggingClassifier(estimator=estimators[estimator], n_jobs=-1)
        model_id.append(estimator)

        for metrica in metricas:
            scoring = cross_val_score(bagging, x_instances, y_instances, scoring=metrica, cv=5)
            performance[i].append(round(scoring.mean(), 2))

        model = bagging.fit(x_instances, y_instances)
        joblib.dump(model, r'' + name_model_file)

    print("Modelos Bagging construidos")
    results = pd.DataFrame(performance, columns=metricas, index=model_id)

    with open(name_evaluation_file, 'w') as f:
        utils.create_txt_report(results, model_id, f)

    name_graph = source_folder + '/bagging_barplot.png'
    title = 'Evaluación del modelo Bagging'
    xlabel = 'Estimadores'
    ylabel = 'Valor de las métricas'
    utils.plot_results(results, name_graph, title, xlabel, ylabel)
