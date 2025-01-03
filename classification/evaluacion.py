import glob
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

escenario_1 = 'C:/Users/HP/Documents/TESIS/software/Multiclasificador/classification/files/escenario_de_prueba_1.csv'
escenario_2 = 'C:/Users/HP/Documents/TESIS/software/Multiclasificador/classification/files/escenario_de_prueba_2.csv'
escenario_3 = 'C:/Users/HP/Documents/TESIS/software/Multiclasificador/classification/files/escenario_de_prueba_3.csv'
escenario_4 = 'C:/Users/HP/Documents/TESIS/software/Multiclasificador/classification/files/escenario_de_prueba_4.csv'
escenario_5 = 'C:/Users/HP/Documents/TESIS/software/Multiclasificador/classification/files/escenario_de_prueba_5.csv'

ensembles = ['bagging', 'adaboosting', 'voting', 'stacking']
bagging_models='C:/Users/HP/Documents/TESIS/software/Multiclasificador/bagging/models_and_evaluation/models/*.pickle'
adaboosting_models='C:/Users/HP/Documents/TESIS/software/Multiclasificador/adaboosting/models_and_evaluation/models' \
                    '/*.pickle'
voting_models = 'C:/Users/HP/Documents/TESIS/software/Multiclasificador/voting/models_and_evaluation/models/*.pickle'
stacking_models='C:/Users/HP/Documents/TESIS/software/Multiclasificador/stacking/models_and_evaluation/models/*.pickle'
bagg_prefix = 'Bag-'
ada_prefix = 'Ada-'
vot_prefix = 'Vot-'
stack_prefix = 'Stk-'


def create_dataframe(model_paths, prefix, drop_string):
    index = []
    for path_model in glob.glob(model_paths):
        file_name = path_model.split('\\')[-1]
        label = file_name.replace(drop_string, '')
        index.append(prefix + label)
    results = pd.DataFrame(columns=[1, 2, 3, 4, 5], index=index)

    return results


def evaluate_models_and_plot():
    names_1 = ['n_rows', "f1", "f2", "f3", "f4", "l1", "l2", "l3", "n1", "n2", "n3", "n4", "t1", "t2", "t3", "t4", "c1",
               "c2", 'n_bots']
    names_2 = ['n_rows', "f1", "f2", "f3", "f4", "l1", "l2", "l3", "n1", "n2", "n3", "n4", "t1", "t2", "t3", "t4",
               "c1", "c2"]

    df_1 = pd.read_csv(escenario_1, sep=';', names=names_1)
    df_2 = pd.read_csv(escenario_2, sep=';', names=names_2)
    df_3 = pd.read_csv(escenario_3, sep=';', names=names_2)
    df_4 = pd.read_csv(escenario_4, sep=';', names=names_2)
    df_5 = pd.read_csv(escenario_5, sep=';', names=names_2)

    meta_features = ["f1", "f2", "f3", "f4", "l1", "l2", "l3", "n1", "n2", "n3", "n4", "t1", "t2", "t3", "t4", "c1",
                     "c2"]
    df_3['c2'].replace([np.inf, -np.inf], 0, inplace=True)
    df_5['c2'].replace([np.inf, -np.inf], 0, inplace=True)

    y_true_1 = []
    y_true_2 = []
    y_true_3 = []

    for j in range(0, 200, 1):
        y_true_1.append(1)
        y_true_2.append(0)
        y_true_3.append(1)
    df_1['label'] = y_true_1
    df_2['label'] = y_true_2
    df_3['label'] = y_true_3
    df_4['label'] = y_true_1
    df_5['label'] = y_true_1
    dataframes = [df_1, df_2, df_3, df_4, df_5]

    results_bagging = create_dataframe(bagging_models, bagg_prefix, '-bagging.pickle')
    for path_model in glob.glob(bagging_models):
        file_name = path_model.split('\\')[-1]
        label = file_name.replace('-bagging.pickle', '')
        exactitud = []
        for i, dataframe, in enumerate(dataframes):
            X = np.array(dataframe[meta_features])
            model = joblib.load(path_model)
            y_pred = model.predict(X)
            exactitud.append(round(accuracy_score(dataframe['label'], y_pred), 3))
        results_bagging.loc[bagg_prefix + label] = exactitud

    results_adabbosting = create_dataframe(adaboosting_models, ada_prefix, '-adaboosting.pickle')
    for path_model in glob.glob(adaboosting_models):
        file_name = path_model.split('\\')[-1]
        label = file_name.replace('-adaboosting.pickle', '')
        exactitud = []
        for i, dataframe, in enumerate(dataframes):
            X = np.array(dataframe[meta_features])
            model = joblib.load(path_model)
            y_pred = model.predict(X)
            exactitud.append(round(accuracy_score(dataframe['label'], y_pred), 3))
        results_adabbosting.loc[ada_prefix + label] = exactitud

    results_voting = create_dataframe(voting_models, vot_prefix, '-voting.pickle')
    for path_model in glob.glob(voting_models):
        file_name = path_model.split('\\')[-1]
        label = file_name.replace('-voting.pickle', '')
        exactitud = []
        for i, dataframe, in enumerate(dataframes):
            X = np.array(dataframe[meta_features])
            model = joblib.load(path_model)
            y_pred = model.predict(X)
            exactitud.append(round(accuracy_score(dataframe['label'], y_pred), 3))
        results_voting.loc[vot_prefix + label] = exactitud

    results_stacking = create_dataframe(stacking_models, stack_prefix, '-stacking.pickle')
    for path_model in glob.glob(stacking_models):
        file_name = path_model.split('\\')[-1]
        label = file_name.replace('-stacking.pickle', '')
        exactitud = []
        for i, dataframe, in enumerate(dataframes):
            X = np.array(dataframe[meta_features])
            model = joblib.load(path_model)
            y_pred = model.predict(X)
            exactitud.append(round(accuracy_score(dataframe['label'], y_pred), 3))
        results_stacking.loc[stack_prefix + label] = exactitud

    file = open('./results_dataframe.txt', 'w')
    file.write('bagging')
    file.write(str(results_bagging))
    file.write('\n')
    file.write('Adaboosting')
    file.write(str(results_adabbosting))
    file.write('\n')
    file.write('Voting')
    file.write(str(results_voting))
    file.write('\n')
    file.write('stacking')
    file.write(str(results_stacking))
    file.write('\n')