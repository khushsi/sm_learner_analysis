import copy
import os
import time

import pandas as pd

from experiments.config import constants
from experiments.model.models import LFA_R
from experiments.model.preprocessor import DataProcessor, Concepts

start_time = time.time()

if __name__ == '__main__':

    data_folder = "data/"
    Discretization_type = 'DiscByCollegeNormal300wpm'
    interaction_data_file = 'data/interaction_data/data_{dataname}.pkl'
    student_split_percent = .7
    interactions_split_percent = .3
    no_of_folds = 10
    student_interaction_file = "IR_17Fall_Preproc2.csv"
    concept_file_folder = 'concept_files/gold/'

    df_interactions = pd.read_csv(data_folder + student_interaction_file)
    if constants.debug:
        df_interactions = df_interactions.head(6000)

    df_interactions[constants.item_field] = ""
    df_interactions[constants.interaction_type] = constants.interaction_type_read
    df_interactions[constants.performance_field] = 0

    for index, row in df_interactions.iterrows():
        if row['is_question'] == 1:
            df_interactions.loc[index, constants.interaction_type] = constants.interaction_type_quiz
            df_interactions.loc[index, constants.item_field] = 'q' + str(
                int(df_interactions.loc[index][constants.questionid_field]))
            df_interactions.loc[index, constants.performance_field] = row['is_correct']
        else:
            df_interactions.loc[index, constants.item_field] = str(df_interactions.loc[index][constants.pageid_field])
            df_interactions.loc[index, constants.performance_field] = row[Discretization_type]
            df_interactions.loc[index, constants.read_behaviour_field] = row[Discretization_type]

    int_data = DataProcessor(df_interactions)
    concept_dictionary = {}

    for concept_file in os.listdir(data_folder + concept_file_folder):
        concept_file_name = concept_file.split("\.")[0]
        for topn in [-1, 5, 10, 15, 20, 25, 30, 40]:
            concept_name = concept_file_name + " : " + str(topn)
            concept_obj = Concepts(name=concept_name,
                                   concept_file=data_folder + concept_file_folder + concept_file, weighted=True,
                                   topn=topn)
            concept_dictionary[concept_name] = copy.copy(concept_obj)

    interaction_folds = int_data.studentwise_interaction__type_folds(no_of_folds=no_of_folds,
                                                                     split_by_student=student_split_percent,
                                                                     split_by_interaction=interactions_split_percent)

    for concept_list_name in concept_dictionary:
        concept_obj = concept_dictionary[concept_list_name]
        int_data.get_concept_distribution(dconcepts=concept_obj)
        int_folds = int_data.get_factor_analysis_interaction_folds_activities(interaction_folds, constants.fields)
        int_con_attempts = int_data.get_interaction_concept_attempts_activities(concept_obj)
        # pfa_mod = LFA_R(concept_obj.concept_list, int_data.studentList, name="lfa" + concept_obj.conceptlistname)
        # pfa_mod.setFolds(int_folds, int_con_attempts)
        # pfa_mod.fitFolds()
        # pfa_mod.predictFolds()
        # pfa_mod.printResult()

        # print(" Top N "+ concept_list_name)

        pfa_mod = LFA_R(concept_obj.concept_list, int_data.studentList, name="lfa_" + concept_obj.conceptlistname,
                        is_reading_behaviour=False)
        pfa_mod.setFolds(int_folds, int_con_attempts)
        pfa_mod.fitFolds()
        pfa_mod.predictFolds()
        pfa_mod.printResult()
        # pfa_mod.fitAIC()

        pfa_mod = LFA_R(concept_obj.concept_list, int_data.studentList, name="lfa_r_" + concept_obj.conceptlistname,
                        is_reading_behaviour=True)
        pfa_mod.setFolds(int_folds, int_con_attempts)
        pfa_mod.fitFolds()
        pfa_mod.predictFolds()
        pfa_mod.printResult()

        pfa_mod = LFA_R(concept_obj.concept_list, int_data.studentList, name="pfa_" + concept_obj.conceptlistname,
                        is_reading_behaviour=False, is_pfa=True)
        pfa_mod.setFolds(int_folds, int_con_attempts)
        pfa_mod.fitFolds()
        pfa_mod.predictFolds()
        pfa_mod.printResult()
        # pfa_mod.fitAIC()

        pfa_mod = LFA_R(concept_obj.concept_list, int_data.studentList, name="pfa_r_" + concept_obj.conceptlistname,
                        is_reading_behaviour=True, is_pfa=True)
        pfa_mod.setFolds(int_folds, int_con_attempts)
        pfa_mod.fitFolds()
        pfa_mod.predictFolds()
        pfa_mod.printResult()

        pfa_mod = LFA_R(concept_obj.concept_list, int_data.studentList, name="pfa_rs_" + concept_obj.conceptlistname,
                        is_reading_behaviour=True, is_pfa=True, is_read_skim=True)
        pfa_mod.setFolds(int_folds, int_con_attempts)
        pfa_mod.fitFolds()
        pfa_mod.predictFolds()
        pfa_mod.printResult()
        # pfa_mod.fitAIC()

    print("end")
