import copy
import os
import time

import pandas as pd

from experiments.config import constants
from experiments.model.models import PFA
from experiments.model.preprocessor import DataProcessor, Concepts

start_time = time.time()

if __name__ == '__main__':

    data_folder = "data/"
    Discretization_type = 'DiscPerDoc'
    interaction_data_file = 'data/interaction_data/data_{dataname}.pkl'
    student_split_percent = .5
    interactions_split_percent = .5
    student_interaction_file = "IR_17Fall_Preproc2.csv"
    concept_file_folder = 'concept_files/'

    df_interactions = pd.read_csv(data_folder + student_interaction_file)
    if constants.debug:
        df_interactions = df_interactions.head(2000)

    df_interactions[constants.item_field] = ""
    df_interactions[constants.interaction_type] = 'Read'
    for index, row in df_interactions.iterrows():
        if row['is_question'] == 1:
            df_interactions[constants.interaction_type] = 'Quiz'
            df_interactions.loc[index, constants.item_field] = 'q' + str(
                int(df_interactions.loc[index][constants.questionid_field]))
        else:
            df_interactions.loc[index, constants.item_field] = str(df_interactions.loc[index][constants.pageid_field])

    int_data = DataProcessor(df_interactions)
    concept_dictionary = {}

    for concept_file in os.listdir(data_folder + concept_file_folder):
        concept_name = concept_file.split("\.")[0]
        for topn in [1]:  # , 2, 5, 10, 15,20,-1]:
            concept_name = concept_name + " : " + str(topn)
            concept_obj = Concepts(name=concept_name,
                                   concept_file=data_folder + concept_file_folder + concept_file, weighted=True, topn=5)
            concept_dictionary[concept_name] = copy.copy(concept_obj)

    interaction_folds = int_data.studentwise_interaction_folds(no_of_folds=2, split_by_student=0.5,
                                                               split_by_interaction=.5)

    for concept_obj in concept_dictionary:
        con_folds = int_data.get_factor_analysis_input(concept_dictionary[concept_obj], interaction_folds,
                                                       constants.fields)
        pfa_mod = PFA(name="pfa" + concept_obj.conceptlistname)




    print("end")
