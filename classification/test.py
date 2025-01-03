from classification.pruebas import Pruebas
from classification.evaluacion import evaluate_models_and_plot

escenario = 'D:/Python Scripts/preprocessData/smote/3/minmax/3.minmax_smote.pickle'
file_data_pf = 'C:/Users/HP/Documents/TESIS/software/datos/file_clasf_pf.pckl'


def models_tests():
    # pruebas = Pruebas(escenario, file_data_pf)
    # pruebas.run_test_scenario_1()
    # pruebas.run_test_scenario_2()
    # pruebas.run_test_scenario_3()
    # pruebas.run_test_scenario_4()
    # pruebas.run_test_scenario_5()
    evaluate_models_and_plot()
