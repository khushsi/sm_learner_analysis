import time

import numpy as np
import pandas as pd

from experiments.config import constants
from experiments.model.models import LFA
from experiments.model.preprocessor import DataProcessor

start_time = time.time()

if __name__ == '__main__':

    data_folder = "data_oli/"
    Discretization_type = 'DiscByCollegeNormal300wpm'
    interaction_data_file = 'data/interaction_data/data_{dataname}.pkl'
    student_split_percent = .5
    interactions_split_percent = .5
    no_of_folds = 10
    student_interaction_file = "merged_step_read_file4_160.csv"

    df_interactions = pd.read_csv(data_folder + student_interaction_file)
    if constants.debug:
        df_interactions = df_interactions.head(6000)

    df_interactions[constants.item_field] = ""
    df_interactions[constants.interaction_type] = 'Read'
    df_interactions[constants.performance_field] = 0

    df_interactions[constants.performance_field] = np.where(df_interactions['correct'] == 'correct', 1, 0)
    ## dummy column
    df_interactions[constants.group_field] = 1

    int_data = DataProcessor(df_interactions)
    concept_dictionary = {}



    interaction_folds = int_data.studentwise_interaction_folds(no_of_folds=no_of_folds,
                                                               split_by_student=student_split_percent,
                                                               split_by_interaction=interactions_split_percent)

    int_folds = int_data.get_factor_analysis_interaction_folds(interaction_folds, constants.fields)

    pfa_mod = LFA(int_data.studentList, name="lfa")
    pfa_mod.setFolds_inline(int_folds)
    # pfa_mod.fitAIC()
    pfa_mod.fitFolds()
    pfa_mod.predictFolds()
    pfa_mod.printResult()

    print("end")
