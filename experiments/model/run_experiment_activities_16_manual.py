import copy
import os
import time

import pandas as pd

from experiments.config import constants
from experiments.model.models import LFA_R
from experiments.model.preprocessor import DataProcessor, Concepts

start_time = time.time()


def run_experiment(student_split_percent,
                   interactions_split_percent,
                   no_of_folds,
                   int_data,
                   Discretization_type,
                   concept_dictionary,
                   is_reading_behaviour=False,
                   is_pfa=False,
                   is_read_skim=False,
                   ):
    print("student split", student_split_percent)
    print("interaction split", interactions_split_percent)
    print("Discretization Type", Discretization_type)
    print("No of Folds", no_of_folds)
    print("student_interaction", student_interaction_file)
    if is_pfa:
        model = "pfa "
    else:
        model = "lfa "
    if is_reading_behaviour:
        model += " reading "
    else:
        model += " no reading "
    if is_read_skim:
        model += "read attempts "
    else:
        model += "read mode "

    for concept_list_name in concept_dictionary:
        concept_obj = concept_dictionary[concept_list_name]
        int_data.get_concept_distribution(dconcepts=concept_obj)
        int_folds = int_data.get_factor_analysis_interaction_folds_activities(interaction_folds, constants.fields)
        int_con_attempts = int_data.get_interaction_concept_attempts_activities(concept_obj)
        # concept_obj.concept_list = [concept for concept in concept_obj.concept_list if concept != 'debug']
        pfa_mod = LFA_R(concept_obj.concept_list,
                        int_data.studentList,
                        name=model + concept_obj.conceptlistname,
                        is_reading_behaviour=is_reading_behaviour,
                        is_pfa=is_pfa,
                        is_read_skim=is_read_skim)

        pfa_mod.setFolds(int_folds, int_con_attempts)
        pfa_mod.fitFolds()
        pfa_mod.predictFolds()
        pfa_mod.printResult()

    print("end")


def processInteractionData(student_interaction_file):
    df_interactions = pd.read_csv(student_interaction_file)
    # if constants.debug:
    #     df_interactions = df_interactions.head(6000)

    df_interactions[constants.item_field] = ""
    df_interactions[constants.interaction_type] = constants.interaction_type_read
    df_interactions[constants.performance_field] = 0

    for index, row in df_interactions.iterrows():
        if row['que_id'] > 0:
            df_interactions.loc[index, constants.interaction_type] = constants.interaction_type_quiz
            df_interactions.loc[index, constants.item_field] = 'q' + str(
                int(df_interactions.loc[index][constants.questionid_field]))
            df_interactions.loc[index, constants.performance_field] = int(row['is_correct'])
        else:
            df_interactions.loc[index, constants.item_field] = str(
                int(df_interactions.loc[index][constants.pageid_field]))
            df_interactions.loc[index, constants.performance_field] = int(row[Discretization_type])
            df_interactions.loc[index, constants.read_behaviour_field] = row[Discretization_type]

    # lquestionid68 = [307, 308, 309, 310, 311, 312, 313, 314, 315, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336]
    # lquestionid = [316 ,317 ,318 ,319 ,320 ,321 ,322 ,323 ,324 ,385 ,386 ,387 ,388 ,389 ,390 ,391 ,392 ,393 ,394 ,395 ,396 ,397 ,398 ,399 ,400 ,401 ,402 ,403 ,404 ,405 ,406 ,407 ,408 ,409 ,410 ,411 ,412 ,413 ,414 ,415]
    # print(len(df_interactions))
    # df_temp_interactions = df_interactions[df_interactions[constants.interaction_type] == constants.interaction_type_read]
    # print(len(df_temp_interactions))
    # print(len(df_interactions[df_interactions[constants.interaction_type] == constants.interaction_type_quiz]))
    # df_interactions_quiz = df_interactions[(df_interactions[constants.questionid_field].isin(lquestionid)) ]
    # print(len(df_interactions_quiz))
    #
    # df_interactions = pd.concat([df_interactions_quiz,df_temp_interactions])
    # print(len(df_interactions))

    int_data = DataProcessor(df_interactions)
    return int_data


def getConcepts(concept_file, topn_list):
    concept_dictionary = {}

    concept_file_name = concept_file.split("\.")[0]
    for topn in topn_list:
        concept_name = concept_file_name + " : " + str(topn)
        concept_obj = Concepts(name=concept_name,
                               concept_file=concept_file, weighted=True,
                               topn=topn)
        concept_dictionary[concept_name] = copy.copy(concept_obj)
    return concept_dictionary


if __name__ == '__main__':

    data_folder = "data/"
    concept_file_folder = 'concept_files/kc/'
    Discretization_type = 'DiscByNormalSpeed'
    interaction_data_file = 'data/interaction_data/data_{dataname}.pkl'

    student_interaction_file = "IR_16Spring_Preprocess1.csv"
    print(concept_file_folder)
    interactions = processInteractionData(data_folder + student_interaction_file)
    topn_list = [5]  # [-1,5,10,15,20,25,30,35,40,45,50]
    concept_dictionary = {}
    for concept_file in os.listdir(data_folder + concept_file_folder):
        concept_dictionary.update(getConcepts(data_folder + concept_file_folder + concept_file, topn_list=topn_list))

    for student_split_percent in [0.5, 0.6, 0.7, 0.8]:
        for interactions_split_percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for no_of_folds in [10]:
                for Discretization_type in ['DiscByNormalSpeed']:
                    interaction_folds = interactions.studentwise_interaction_folds(no_of_folds=no_of_folds,
                                                                                   split_by_student=student_split_percent,
                                                                                   split_by_interaction=interactions_split_percent,
                                                                                   type=[
                                                                                       constants.interaction_type_quiz])
                    # run_experiment(student_split_percent=student_split_percent,
                    #                interactions_split_percent=interactions_split_percent,
                    #                no_of_folds=no_of_folds,
                    #                int_data=interactions,
                    #                Discretization_type=Discretization_type,
                    #                is_reading_behaviour=False,
                    #                is_pfa=False,
                    #                is_read_skim=False,
                    #                concept_dictionary=concept_dictionary)
                    #
                    # run_experiment(student_split_percent=student_split_percent,
                    #                interactions_split_percent=interactions_split_percent,
                    #                no_of_folds=no_of_folds,
                    #                int_data=interactions,
                    #                Discretization_type=Discretization_type,
                    #                is_reading_behaviour=True,
                    #                is_pfa=False,
                    #                is_read_skim=False,
                    #                concept_dictionary=concept_dictionary)
                    #
                    # run_experiment(student_split_percent=student_split_percent,
                    #                interactions_split_percent=interactions_split_percent,
                    #                no_of_folds=no_of_folds,
                    #                int_data=interactions,
                    #                Discretization_type=Discretization_type,
                    #                is_reading_behaviour=True,
                    #                is_pfa=False,
                    #                is_read_skim=True,
                    #                concept_dictionary=concept_dictionary)

                    # run_experiment(student_split_percent=student_split_percent,
                    #                interactions_split_percent=interactions_split_percent,
                    #                no_of_folds=no_of_folds,
                    #                int_data=interactions,
                    #                Discretization_type=Discretization_type,
                    #                is_reading_behaviour=False,
                    #                is_pfa=True,
                    #                is_read_skim=True,
                    #                concept_dictionary=concept_dictionary)

                    # run_experiment(student_split_percent=student_split_percent,
                    #                interactions_split_percent=interactions_split_percent,
                    #                no_of_folds=no_of_folds,
                    #                int_data=interactions,
                    #                Discretization_type=Discretization_type,
                    #                is_reading_behaviour=True,
                    #                is_pfa=True,
                    #                is_read_skim=False,
                    #                concept_dictionary=concept_dictionary)

                    run_experiment(student_split_percent=student_split_percent,
                                   interactions_split_percent=interactions_split_percent,
                                   no_of_folds=no_of_folds,
                                   int_data=interactions,
                                   Discretization_type=Discretization_type,
                                   is_reading_behaviour=True,
                                   is_pfa=True,
                                   is_read_skim=True,
                                   concept_dictionary=concept_dictionary)
