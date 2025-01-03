import glob
import os
import joblib
from utils.metrics import Metric
from utils import utils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path_file = './design/databases/metricas_calculadas.csv'


def create_bots_subsets(escenario1, escenario2, data_pf, count):
    columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']
    X, y = utils.merge_scenarios(escenario1, escenario2)
    size_list1 = np.random.randint(low=10, high=10000, size=count).tolist()
    for i, size in enumerate(size_list1):
        x_selected, _, y_selected, _ = train_test_split(X,
                                                        y,
                                                        train_size=size,
                                                        shuffle=True,
                                                        random_state=2022
                                                        )
        x_instances, y_instances, y_label = utils.data_labeling(x_selected, y_selected, data_pf)
        metricas = Metric(x_instances, y_label).run_metrics()
        # Calcula las metricas de complejidad del subconjunto de datos
        path_files = './design/databases/conjuntos con bots/B' + str(i + 1) + '.csv'
        bot_subset_label = 1
        data_subset = pd.DataFrame(data=x_instances, columns=columns)
        data_subset['true_label'] = y_instances
        data_subset['pred_label'] = y_label
        data_subset.to_csv(path_or_buf=path_files, sep=';')
        id_subset = 'B' + str(i + 1)

        with open(path_file, 'a') as f:
            utils.sored_data_with_label(id_subset, size, metricas, bot_subset_label, f)


def create_human_subsets(escenario1, escenario2, data_pf, count):
    global x_negatives
    columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']
    X, y = utils.merge_scenarios(escenario1, escenario2)
    size_list2 = np.random.randint(low=10, high=10000, size=count).tolist()
    for b, size in enumerate(size_list2):
        x_negatives = []
        true_label = []
        pred_label = []
        x_selected, _, y_selected, _ = train_test_split(X,
                                                        y,
                                                        train_size=size,
                                                        shuffle=True,
                                                        random_state=2022
                                                        )
        x_instances, y_instances, y_label = utils.data_labeling(x_selected, y_selected, data_pf)
        for i, j, k in zip(x_instances, y_instances, y_label):
            if k == 0:
                x_negatives.append(i)
                true_label.append(j)
                pred_label.append(k)
        metricas = Metric(np.array(x_negatives), pred_label).run_metrics()
        data_subset = pd.DataFrame(data=x_negatives, columns=columns)
        data_subset['true_label'] = true_label
        data_subset['pred_label'] = pred_label
        # Calcula las metricas de complejidad del subconjunto de datos
        path_files = './design/databases/conjuntos sin bots/H' + str(b + 1) + '.csv'
        data_subset.to_csv(path_or_buf=path_files, sep=';')

        human_subset_label = 0
        id_subset = 'H' + str(b + 1)
        size = np.array(x_negatives).shape[0]
        del x_negatives
        del true_label
        del pred_label
        with open(path_file, 'a') as f:
            utils.sored_data_with_label(id_subset, size, metricas, human_subset_label, f)


class homogeneo:
    def __init__(self, models, data):
        self.data = data
        self.models = models
        self.columns = ['ID', 'rows', "f1", "f2", "f3", "f4", "l1", "l2", "l3", "n1", "n2", "n3", "n4", "t1", "t2",
                        "t3", "t4", "c1", "c2", 'label']
        self.meta_features = ["f1", "f2", "f3", "f4", "l1", "l2", "l3", "n1", "n2", "n3", "n4", "t1", "t2",
                              "t3", "t4", "c1", "c2"]
        self.df = pd.read_csv(path_file, sep=';', names=self.columns)
        self.X_test = self.df[self.meta_features]
        self.true_label = self.df['label']

    def evaluar(self):
        for ensemble in self.models:
            performance = []
            list_models = []
            path_models = 'C:/Users/HP/Documents/TESIS/software/Multiclasificador/' + ensemble + \
                          '/models_and_evaluation/models/*.pickle'
            for i, model in enumerate(glob.glob(path_models)):
                file_name = model.split('\\')[-1]
                if ensemble == 'bagging':
                    label = file_name.replace('ging.pickle', '')
                    list_models.append(label)
                elif ensemble == 'adaboosting':
                    label = file_name.replace('boosting.pickle', '')
                    list_models.append(label)

                model_extracted = joblib.load(model)
                pred_label = model_extracted.predict(self.X_test)
                performance.append([])

                for metrica in utils.metricas.keys():
                    performance[i].append(round(utils.metricas[metrica](self.true_label, pred_label), 2))
            print('Clasificacion con models ' + ensemble + ' realizada')
            results = pd.DataFrame(performance, columns=utils.metricas.keys(), index=list_models)

            name_graph = './design/reports/' + ensemble + '-classification-report.png'
            name_report = './design/reports/' + ensemble + '-classification-report.txt'
            title = 'Evaluación del multiclasificador ' + ensemble
            xlabel = 'Subconjuntos de clasificadores'
            ylabel = 'Valor de las mátricas'
            utils.plot_results(results, name_graph, title, xlabel, ylabel)

            with open(name_report, 'w') as f:
                f.write(str(results))
            results.drop(columns=utils.metricas.keys(), axis=1)


class hibrido:
    def __init__(self, models, data):
        self.performance = []
        self.data = data
        self.models = models
        self.columns = ['ID', 'rows', "f1", "f2", "f3", "f4", "l1", "l2", "l3", "n1", "n2", "n3", "n4", "t1", "t2",
                        "t3", "t4", "c1", "c2", 'label']
        self.meta_features = ["f1", "f2", "f3", "f4", "l1", "l2", "l3", "n1", "n2", "n3", "n4", "t1", "t2",
                              "t3", "t4", "c1", "c2"]
        self.df = pd.read_csv(path_file, sep=';', names=self.columns)
        self.X_test = self.df[self.meta_features]
        self.true_label = self.df['label']

    def evaluar(self):
        for ensemble in self.models:
            performance = []
            list_models = []
            path_models = 'C:/Users/HP/Documents/TESIS/software/Multiclasificador/' + ensemble + \
                          '/models_and_evaluation/models/*.pickle'
            for i, model in enumerate(glob.glob(path_models)):
                file_name = model.split('\\')[-1]
                label = file_name.replace('.pickle', '')
                if ensemble == 'voting':
                    list_models.append(int(label.replace('Vot-', '')))
                elif ensemble == 'stacking':
                    list_models.append(int(label.replace('Stk-', '')))

                model_extracted = joblib.load(model)
                pred_label = model_extracted.predict(self.X_test)
                performance.append([])

                for metrica in utils.metricas.keys():
                    performance[i].append(round(utils.metricas[metrica](self.true_label, pred_label), 2))
            print('Clasificacion con models ' + ensemble + ' realizada')
            results = pd.DataFrame(performance, columns=utils.metricas.keys(), index=list_models)
            results1 = results.sort_index()
            if ensemble == 'voting':
                index_order = ['Vot-1', 'Vot-2', 'Vot-3', 'Vot-4', 'Vot-5', 'Vot-6', 'Vot-7', 'Vot-8', 'Vot-9',
                               'Vot-10', 'Vot-11', 'Vot-12', 'Vot-13', 'Vot-14', 'Vot-15']
                results1.index = index_order

            elif ensemble == 'stacking':
                index_order = ['Stk-1', 'Stk-2', 'Stk-3', 'Stk-4', 'Stk-5', 'Stk-6', 'Stk-7', 'Stk-8', 'Stk-9',
                               'Stk-10', 'Stk-11', 'Stk-12', 'Stk-13', 'Stk-14', 'Stk-15']
                results1.index = index_order

            name_graph = './design/reports/' + ensemble + '-classification-report.png'
            name_report = './design/reports/' + ensemble + '-classification-report.txt'
            title = 'Evaluación del multiclasificador ' + ensemble
            xlabel = 'Subconjuntos de clasificadores'
            ylabel = 'Valor de las mátricas'
            utils.plot_results(results1, name_graph, title, xlabel, ylabel)

            with open(name_report, 'w') as f:
                f.write(str(results))
            results.drop(columns=utils.metricas.keys(), axis=1)


def run_design_experiments(escenario1, escenario2, homogeneos_list, hibridos_list, data_file_path, data_pf):
    subsets_count = 100  # Se crearán 100 subconjunto de datos para cada clase
    if os.path.exists(path_file):
        os.remove(path_file)

    create_bots_subsets(escenario1, escenario2, data_pf, subsets_count)
    create_human_subsets(escenario1, escenario2, data_pf, subsets_count)

    homogeneo(homogeneos_list, data_file_path).evaluar()
    hibrido(hibridos_list, data_file_path).evaluar()
