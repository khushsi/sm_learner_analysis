import copy
import os
import time

import pandas as pd

from experiments.config import constants
from experiments.model.models import LFA
from experiments.model.preprocessor import DataProcessor, Concepts

start_time = time.time()

if __name__ == '__main__':

    data_folder = "data/"
    Discretization_type = 'DiscByCollegeNormal300wpm'
    interaction_data_file = 'data/interaction_data/data_{dataname}.pkl'
    student_split_percent = .5
    interactions_split_percent = .96
    no_of_folds = 10
    student_interaction_file = "IR_17Fall_Preproc2.csv"
    concept_file_folder = 'concept_files/lda/'

    df_interactions = pd.read_csv(data_folder + student_interaction_file)
    if constants.debug:
        df_interactions = df_interactions.head(600)

    df_interactions[constants.item_field] = ""
    df_interactions[constants.interaction_type] = 'Read'
    df_interactions[constants.performance_field] = 0

    for index, row in df_interactions.iterrows():
        if row['is_question'] == 1:
            df_interactions.loc[index, constants.interaction_type] = 'Quiz'
            df_interactions.loc[index, constants.item_field] = 'q' + str(
                int(df_interactions.loc[index][constants.questionid_field]))
            df_interactions.loc[index, constants.performance_field] = row['is_correct']
        else:
            df_interactions.loc[index, constants.item_field] = str(df_interactions.loc[index][constants.pageid_field])
            df_interactions.loc[index, constants.performance_field] = row[Discretization_type]

    int_data = DataProcessor(df_interactions)
    concept_dictionary = {}

    for concept_file in os.listdir(data_folder + concept_file_folder):

        for topn in [5]:
            concept_name = concept_file.split("\.")[0]
            concept_name = concept_name + " : " + str(topn)
            concept_obj = Concepts(name=concept_name,
                                   concept_file=data_folder + concept_file_folder + concept_file, weighted=True,
                                   topn=topn)
            concept_dictionary[concept_name] = copy.copy(concept_obj)

    interaction_folds = int_data.studentwise_interaction_folds(no_of_folds=no_of_folds,
                                                               split_by_student=student_split_percent,
                                                               split_by_interaction=interactions_split_percent)

    for concept_list_name in concept_dictionary:
        concept_obj = concept_dictionary[concept_list_name]
        int_folds = int_data.get_factor_analysis_interaction_folds(interaction_folds, constants.fields)
        int_con_attempts = int_data.get_interaction_concept_attempts(concept_obj)
        pfa_mod = LFA(concept_obj.concept_list, int_data.studentList, name="lfa" + concept_obj.conceptlistname)

        pfa_mod.setFolds(int_folds, int_con_attempts)
        pfa_mod.fitFolds()
        pfa_mod.predictFolds()
        pfa_mod.printResult()

    print("end")
