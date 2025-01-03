from utils import utils as utils
import numpy as np

report_file_1 = 'C:/Users/HP/Documents/TESIS/software/Multiclasificador/classification/files/escenario_de_prueba_1.csv'
report_file_2 = 'C:/Users/HP/Documents/TESIS/software/Multiclasificador/classification/files/escenario_de_prueba_2.csv'
report_file_3 = 'C:/Users/HP/Documents/TESIS/software/Multiclasificador/classification/files/escenario_de_prueba_3.csv'
report_file_4 = 'C:/Users/HP/Documents/TESIS/software/Multiclasificador/classification/files/escenario_de_prueba_4.csv'
report_file_5 = 'C:/Users/HP/Documents/TESIS/software/Multiclasificador/classification/files/escenario_de_prueba_5.csv'


class Pruebas:
    def __init__(self, data_test, data_pf):
        self.file_data_pf = data_pf
        self.escenario = data_test
        self.x_instances, self.y_instances = utils.extract_pickle_file(self.escenario)
        self.X, self.y_label = utils.data_labeling(self.x_instances, self.file_data_pf)

    def run_test_scenario_1(self):
        for rows in range(50, 10050, 50):
            metricas, y_subset = utils.data_set_metrics(self.X, self.y_label, rows)
            # Cuenta el número de bots que hay en el subconjunto de datos
            n_bots = 0
            for j in y_subset:
                if j == 1:
                    n_bots += 1

            # Calcula las metricas de complejidad del subconjunto de datos
            with open(report_file_1, 'a') as f:
                utils.sored_data_with_label_1(rows, metricas, n_bots, f)

    def run_test_scenario_2(self):
        x_negatives = []
        y_label = []
        for i, j in zip(self.X, self.y_label):
            if j == 0.0:
                x_negatives.append(i)
                y_label.append(j)

        X = np.array(x_negatives)

        for rows in range(50, 10050, 50):
            metricas, y_subset = utils.data_set_metrics(X, y_label, rows)
            with open(report_file_2, 'a') as f:
                utils.sored_data_with_label(rows, metricas, f)

    def run_test_scenario_3(self):
        x_positives = []
        y_label = []
        for i, j in zip(self.X, self.y_label):
            if j == 1.0:
                x_positives.append(i)
                y_label.append(j)

        X = np.array(x_positives)
        for rows in range(50, 10050, 50):
            metricas, y_subset = utils.data_set_metrics(X, y_label, rows)
            with open(report_file_3, 'a') as f:
                utils.sored_data_with_label(rows, metricas, f)

    def run_test_scenario_4(self):
        new_label = []
        for i in range(0, len(self.y_label)):
            new_label.append(0)

        for rows in range(50, 10050, 50):
            metricas, y_subset = utils.data_set_metrics(self.X, new_label, rows)

            # Calcula las metricas de complejidad del subconjunto de datos
            with open(report_file_4, 'a') as f:
                utils.sored_data_with_label(rows, metricas, f)

    def run_test_scenario_5(self):
        new_label = []
        for i in range(0, len(self.y_label)):
            new_label.append(1)
        for rows in range(50, 10050, 50):
            metricas, y_subset = utils.data_set_metrics(self.X, new_label, rows)
            # Calcula las metricas de complejidad del subconjunto de datos
            with open(report_file_5, 'a') as f:
                utils.sored_data_with_label(rows, metricas, f)
