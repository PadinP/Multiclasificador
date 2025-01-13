from sklearn.metrics import classification_report
from models.bagging import build_bagging_models
from models.boosting import build_ada_boosting_models
from models.individual import Diversity
from models.stacking import build_stacking_models
from models.voting import build_voting_models
from classification.test import models_tests
from utils.utils import estimators
from design.experiments import run_design_experiments
from design.experiments import homogeneo
import warnings
import pandas as pd
import pickle
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

database = 'files/characterization_database.txt' 
escenario11 = 'files/1.minmax_smote.pickle' 
escenario3 = 'files/0.minmax_smote.pickle' 
data_pf = 'files/file_clasf_pf_bueno.pckl' 
ensambles_homogeneos = ['bagging', 'adaboosting']
ensambles_hibridos = ['voting', 'stacking']
# path_file = './design/databases/metricas_calculadas.csv'
path_file = 'design/databases/metricas_calculadas_2.csv'
path_capturas = 'files/capturas_procesadas'
def main():
    build_bagging_models(estimators, database)
    build_ada_boosting_models(estimators, database)
    diversity = Diversity(estimators, database)
    diversity.diversity_calc()
    subsets = diversity.select_subsets()
    build_voting_models(subsets, database)
    build_stacking_models(subsets, database)
    # # models_tests()

    run_design_experiments(path_capturas,
                           ensambles_homogeneos,
                           ensambles_hibridos,
                           path_file, data_pf)



    # # Cargar el archivo pickle
    # with open(escenario3, 'rb') as file:
    #     data = pickle.load(file)

    # # Convertir los datos a una lista de diccionarios si es necesario
    # # Suponiendo que tus datos son una lista de listas, donde cada sublista representa una fila
    # # Ajusta este paso según la estructura real de tus datos
    # processed_data = [dict(enumerate(row)) for row in data]

    # # Crear un DataFrame de pandas
    # df = pd.DataFrame(processed_data)

    # # Guardar el DataFrame como un archivo CSV
    # df.to_csv('archivo.csv', index=False)



if __name__ == '__main__':
    main()
