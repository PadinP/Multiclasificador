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

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

database = 'C:/Users/HP/Documents/TESIS/software/datos/description_database.txt'
escenario11 = 'D:/Python Scripts/preprocessData/smote/11/minmax/11.minmax_smote.pickle'
escenario3 = 'D:/Python Scripts/preprocessData/smote/3/minmax/3.minmax_smote.pickle'
data_pf = 'C:/Users/HP/Documents/TESIS/software/datos/file_clasf_pf_11.pckl'
path_file = './design/databases/metricas_calculadas.csv'
ensambles_homogeneos = ['bagging', 'adaboosting']
ensambles_hibridos = ['voting', 'stacking']


def main():
    # build_bagging_models(estimators, database)
    # build_ada_boosting_models(estimators, database)
    # diversity = Diversity(estimators, database)
    # diversity.diversity_calc()
    # subsets = diversity.select_subsets()
    # build_voting_models(subsets, database)
    # build_stacking_models(subsets, database)
    # models_tests()

    run_design_experiments(escenario3,
                           escenario11,
                           ensambles_homogeneos,
                           ensambles_hibridos,
                           path_file, data_pf)


if __name__ == '__main__':
    main()
