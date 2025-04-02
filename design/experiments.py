import glob
import os
import joblib
from utils.metrics import Metric
from utils import utils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path_file = './design/databases/metricas_calculadas.csv'
path_file = 'design/databases/metricas_calculadas_2.csv'


def extraer_metricas(directory_path, data_pf):

    # Dictionary to map file names to prefixes
    file_prefix_mapping = {
        '1.minmax_smote.pickle': '50K',
        '2.minmax_smote.pickle': '100K',
        '3.minmax_smote.pickle': '200K'
        
        
    }
    
    for file_name in os.listdir(directory_path):
        # Construct full file path
        file_path = os.path.join(directory_path, file_name)
        
        # Check if the file_name is in our mapping
        if file_name in file_prefix_mapping:
            prefix = file_prefix_mapping[file_name]
            X, y = utils.extract_pickle_file(file_path)
            
            # Usar el conjunto completo de datos
            x_instances, y_instances, y_label = utils.data_labeling(X, y, data_pf)
            bot_subset_label = 1
            
            # calcular métricas y salvarlas en un archivo
            metricas = Metric(x_instances, y_label).run_metrics()
            with open(path_file, '+a') as f:
                utils.sored_data_with_label(prefix, len(X), metricas, bot_subset_label, f)   
        else:
            print(f"Archivo {file_name} no está en el mapeo, se ignora.")




# def create_captura(captura1, captura2, data_pf):
#     path_file = 'design/databases/capturas/metricas_calculadas_2.csv'
#     for i in range(2):
#         # Alternar entre las dos capturas
#         if i == 0:
#             X, y = utils.extract_pickle_file(captura1)
#             prefix = '1K.1'
#         else:
#             X, y = utils.extract_pickle_file(captura2)
#             prefix = '1K.2'
#         # Usar el conjunto completo de datos
#         x_instances, y_instances, y_label = utils.data_labeling(X, y, data_pf)
#         bot_subset_label = 1
#         # print("x_instances (características de los datos):", x_instances)
#         # print("y_instances (etiquetas originales):", y_instances)
#         # print("y_label (etiquetas predichas):", y_label)

#         metricas = Metric(x_instances, y_label).run_metrics()
#         with open(path_file, '+a') as f:
#             utils.sored_data_with_label(prefix,len(X),metricas,bot_subset_label,f)   



# def create_bots_subsets(escenario1, escenario2, data_pf, count):
#     columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']
#     X, y = utils.merge_scenarios(escenario1, escenario2)
#     size_list1 = np.random.randint(low=10, high=10000, size=count).tolist()
#     for i, size in enumerate(size_list1):
#         x_selected, _, y_selected, _ = train_test_split(X,
#                                                         y,
#                                                         train_size=size,
#                                                         shuffle=True,
#                                                         random_state=2022
#                                                         )
#         x_instances, y_instances, y_label = utils.data_labeling(x_selected, y_selected, data_pf)
#         metricas = Metric(x_instances, y_label).run_metrics()
#         # Calcula las metricas de complejidad del subconjunto de datos
#         path_files = './design/databases/conjuntos con bots/B' + str(i + 1) + '.csv'
#         bot_subset_label = 1
#         data_subset = pd.DataFrame(data=x_instances, columns=columns)
#         data_subset['true_label'] = y_instances
#         data_subset['pred_label'] = y_label
#         data_subset.to_csv(path_or_buf=path_files, sep=';')
#         id_subset = 'B' + str(i + 1)

#         with open(path_file, 'a') as f:
#             utils.sored_data_with_label(id_subset, size, metricas, bot_subset_label, f)


# def create_human_subsets(escenario1, escenario2, data_pf, count):
#     global x_negatives
#     columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']
#     X, y = utils.merge_scenarios(escenario1, escenario2)
#     size_list2 = np.random.randint(low=10, high=10000, size=count).tolist()
#     for b, size in enumerate(size_list2):
#         x_negatives = []
#         true_label = []
#         pred_label = []
#         x_selected, _, y_selected, _ = train_test_split(X,
#                                                         y,
#                                                         train_size=size,
#                                                         shuffle=True,
#                                                         random_state=2022
#                                                         )
#         x_instances, y_instances, y_label = utils.data_labeling(x_selected, y_selected, data_pf)
#         for i, j, k in zip(x_instances, y_instances, y_label):
#             if k == 0:
#                 x_negatives.append(i)
#                 true_label.append(j)
#                 pred_label.append(k)
#         metricas = Metric(np.array(x_negatives), pred_label).run_metrics()
#         data_subset = pd.DataFrame(data=x_negatives, columns=columns)
#         data_subset['true_label'] = true_label
#         data_subset['pred_label'] = pred_label
#         # Calcula las metricas de complejidad del subconjunto de datos
#         path_files = './design/databases/conjuntos sin bots/H' + str(b + 1) + '.csv'
#         data_subset.to_csv(path_or_buf=path_files, sep=';')

#         human_subset_label = 0
#         id_subset = 'H' + str(b + 1)
#         size = np.array(x_negatives).shape[0]
#         del x_negatives
#         del true_label
#         del pred_label
#         with open(path_file, 'a') as f:
#             utils.sored_data_with_label(id_subset, size, metricas, human_subset_label, f)


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
        all_predictions = pd.DataFrame()
        prefix = 'C'
        for ensemble in self.models:
            performance = []
            list_models = []
            path_models = f'/home/app/Escritorio/Multiclasificador/{ensemble}/models_and_evaluation/models/*.pickle'
            for i, model in enumerate(glob.glob(path_models)):
                file_name = model.split('/')[-1]
                if ensemble == 'bagging':
                    label = file_name.replace('ging.pickle', '')
                    list_models.append(label)
                elif ensemble == 'adaboosting':
                    label = file_name.replace('boosting.pickle', '')
                    list_models.append(label)

                try:
                    print(f"Cargando modelo: {model}")
                    model_extracted = joblib.load(model)
                    if hasattr(model_extracted, "estimators_"):
                        model_extracted.estimator_ = model_extracted.estimators_[0]
                    pred_label = model_extracted.predict(self.X_test)
                    print(pred_label)
                except ValueError as e:
                    print(f"Error de valor al cargar {model}: {e}")
                    continue
                except AttributeError as e:
                    print(f"Error de atributo con el modelo {model}: {e}")
                    continue
                
                # Añadir predicciones al DataFrame
                all_predictions[f'{ensemble}_{label}'] = pred_label

                performance.append([])
                for metrica in utils.metricas.keys():
                    performance[i].append(round(utils.metricas[metrica](self.true_label, pred_label), 2))

            print(f'Clasificación con modelos {ensemble} realizada')
            results = pd.DataFrame(performance, columns=utils.metricas.keys(), index=list_models)

            name_graph = f'./design/reports/{ensemble}-classification-report.png'
            name_report = f'./design/reports/{ensemble}-classification-report.txt'
            title = f'Evaluación del multiclasificador {ensemble}'
            xlabel = 'Subconjuntos de clasificadores'
            ylabel = 'Valor de las métricas'
            # utils.plot_results(results, name_graph, title, xlabel, ylabel)

            with open(name_report, 'w') as f:
                f.write(str(results))

            results.drop(columns=utils.metricas.keys(), axis=1)

            # Crear un nuevo DataFrame para almacenar el formato deseado 
            formatted_predictions = all_predictions.T 
            formatted_predictions.insert(0, 'Tipo de modelo', formatted_predictions.index)
             # Añadir encabezado adicional 
            header = ['Tipo de modelo'] + [f'{prefix}.{i+1}' 
            for i in range(len(formatted_predictions.columns)-1)] 
            formatted_predictions.columns = header 
            # Guardar las predicciones formateadas en un archivo CSV 
            formatted_predictions.to_csv('design/databases/prediciones/all_predictions_homogeneo.csv', sep=';', index=False)
            
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
        all_predictions = pd.DataFrame()
        prefix = 'C'
        for ensemble in self.models:
            performance = []
            list_models = []
            path_models = f'/home/app/Escritorio/Multiclasificador/{ensemble}/models_and_evaluation/models/*.pickle'
            for i, model in enumerate(glob.glob(path_models)):
                file_name = model.split('/')[-1]
                label = file_name.replace('.pickle', '')
                try:
                    if ensemble == 'voting':
                        if label.startswith('Vot-'):
                            list_models.append(int(label.replace('Vot-', '')))
                        else:
                            print(f"Formato inesperado de label: {label}")
                    elif ensemble == 'stacking':
                        if label.startswith('Stk-'):
                            list_models.append(int(label.replace('Stk-', '')))
                        else:
                            print(f"Formato inesperado de label: {label}")
                except ValueError as e:
                    print(f"Error al convertir label a entero: {e}")

                model_extracted = joblib.load(model)
                pred_label = model_extracted.predict(self.X_test)
                print(pred_label)
                performance.append([])

                # Añadir predicciones al DataFrame
                all_predictions[f'{ensemble}_{label}'] = pred_label

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
            ylabel = 'Valor de las métricas'
            utils.plot_results(results1, name_graph, title, xlabel, ylabel)

            with open(name_report, 'w') as f:
                f.write(str(results))
            results.drop(columns=utils.metricas.keys(), axis=1)

            # Crear un nuevo DataFrame para almacenar el formato deseado 
            formatted_predictions = all_predictions.T 
            formatted_predictions.insert(0, 'Tipo de modelo', formatted_predictions.index)
             # Añadir encabezado adicional 
            header = ['Tipo de modelo'] + [f'{prefix}.{i+1}' 
            for i in range(len(formatted_predictions.columns)-1)] 
            formatted_predictions.columns = header 
            # Guardar las predicciones formateadas en un archivo CSV 
            formatted_predictions.to_csv('design/databases/prediciones/all_predictions_hibrido.csv', sep=';', index=False)

# def run_design_experiments(escenario1, escenario2, homogeneos_list, hibridos_list, data_file_path, data_pf):
#     # subsets_count = 100  # Se crearán 100 subconjunto de datos para cada clase
#     if os.path.exists(path_file):
#         os.remove(path_file)

#     # create_bots_subsets(escenario1, escenario2, data_pf, subsets_count)
#     # create_human_subsets(escenario1, escenario2, data_pf, subsets_count)

#     create_captura(escenario1,escenario2,data_pf)
#     homogeneo(homogeneos_list, data_file_path).evaluar()
#     hibrido(hibridos_list, data_file_path).evaluar()


def run_design_experiments(path_capturas, homogeneos_list, hibridos_list, data_file_path, data_pf):
    if os.path.exists(path_file):
        os.remove(path_file)

    extraer_metricas(path_capturas,data_pf)
    homogeneo(homogeneos_list, data_file_path).evaluar()
    hibrido(hibridos_list, data_file_path).evaluar()