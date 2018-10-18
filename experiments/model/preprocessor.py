import copy
import random as rnd

import numpy as np
import pandas as pd

from experiments.config import constants


class DataProcessor:
    def __init__(self, df_interactions):
        self.__interactions = self.sort_assign_interactionid(df_interactions)
        self.studentList = self.__interactions[constants.student_field].unique()

    def interactions(self):
        return self.__interactions

    def sort_assign_interactionid(self, df_interactions):
        df_interactions = df_interactions.drop(
            df_interactions.columns[df_interactions.columns.str.contains('unnamed', case=False)], axis=1)
        df_interactions = df_interactions.sort_values(
            by=[constants.student_field, constants.starttime_field,
                constants.endtime_field]).reset_index(drop=True)
        df_interactions[constants.interactionid_field] = df_interactions.index

        return df_interactions

    def studentwise_interaction_folds(self, no_of_folds=10,
                                      split_by_student=0.5,
                                      split_by_interaction=.5,
                                      interaction_type=[constants.interaction_type_quiz]):
        # questionid = [307 ,308 ,309 ,310 ,311 ,312 ,313 ,314 ,315 ,325 ,326 ,327 ,328 ,329 ,330 ,331 ,332 ,333 ,334 ,335 ,336]
        folds = []
        fold = None
        df_data = self.interactions()
        df_data = df_data[df_data[constants.interaction_type].isin(interaction_type)]
        # df_data = df_data[df_data[constants.questionid_field].isin(questionid)]
        student_list = df_data[constants.student_field].unique()
        split_student_train_count = int(np.ceil(len(student_list) * split_by_student))

        df_train = None
        df_test = None

        print(student_list)
        for i in range(no_of_folds):
            df_train = None
            df_test = None
            rnd.shuffle(student_list)
            train_student = student_list[0:split_student_train_count]
            test_student = student_list[split_student_train_count + 1:]
            df_train = df_data[(df_data[constants.student_field].isin(train_student))][constants.interactionid_field]
            df_test = df_data[df_data[constants.student_field] == None][constants.interactionid_field]

            for student in test_student:
                df_temp_student = df_data[(df_data[constants.student_field] == student)]
                split_place = int(np.ceil(split_by_interaction * len(df_temp_student)))
                df_train = df_train.append(df_temp_student[0:split_place][constants.interactionid_field])
                df_test = df_test.append(df_temp_student[split_place + 1:][constants.interactionid_field])
            # print(student_list)

            print("Fold No: " + str(i) + " no of interactions - train  : " + str(len(df_train)) + " test : " + str(
                len(df_test)))
            folds.append({'train_interactions': copy.copy(df_train), 'test_interactions': copy.copy(df_test)})

        return folds

    def studentwise_interaction__activity_folds(self, no_of_folds=10, split_by_student=0.5, split_by_interaction=.5):
        folds = []
        fold = None
        df_data = self.interactions()
        df_data = df_data[df_data[constants.interaction_type] == constants.interaction_type_quiz]
        student_list = df_data[constants.student_field].unique()
        split_student_train_count = int(np.ceil(len(student_list) * split_by_student))

        df_train = None
        df_test = None

        # print(student_list)
        for i in range(no_of_folds):
            df_train = None
            df_test = None
            rnd.shuffle(student_list)
            train_student = student_list[0:split_student_train_count]
            test_student = student_list[split_student_train_count + 1:]

            df_train = df_data[(df_data[constants.student_field].isin(train_student))][constants.interactionid_field]
            df_test = df_data[df_data[constants.student_field] == None][constants.interactionid_field]
            for student in test_student:
                df_temp_student = df_data[(df_data[constants.student_field] == student)]
                split_place = int(np.ceil(split_by_interaction * len(df_temp_student)))
                df_train = df_train.append(df_temp_student[0:split_place][constants.interactionid_field])
                df_test = df_test.append(df_temp_student[split_place + 1:][constants.interactionid_field])

            # print(student_list)
            print("Fold No: " + str(i) + " no of interactions - train  : " + str(len(df_train)) + " test : " + str(
                len(df_test)))
            folds.append({'train_interactions': copy.copy(df_train), 'test_interactions': copy.copy(df_test)})

        return folds

    def studentwise_interaction_folds(self,
                                      no_of_folds=10,
                                      split_by_student=0.5,
                                      split_by_interaction=.5,
                                      type=[constants.interaction_type_quiz, constants.interaction_type_read]):
        folds = []
        fold = None
        df_data = self.interactions()
        df_data = df_data[df_data[constants.interaction_type].isin(type)]
        student_list = df_data[constants.student_field].unique()
        split_student_train_count = int(np.ceil(len(student_list) * split_by_student))

        df_train = None
        df_test = None

        # print(student_list)
        for i in range(no_of_folds):
            df_train = None
            df_test = None
            rnd.shuffle(student_list)
            train_student = student_list[0:split_student_train_count]
            test_student = student_list[split_student_train_count + 1:]

            df_train = df_data[(df_data[constants.student_field].isin(train_student))][constants.interactionid_field]
            df_test = df_data[df_data[constants.student_field] == None][constants.interactionid_field]
            for student in test_student:
                df_temp_student = df_data[(df_data[constants.student_field] == student)]
                split_place = int(np.ceil(split_by_interaction * len(df_temp_student)))
                df_train = df_train.append(df_temp_student[0:split_place][constants.interactionid_field])
                df_test = df_test.append(df_temp_student[split_place + 1:][constants.interactionid_field])

            # print(student_list)
            print("Fold No: " + str(i) + " no of interactions - train  : " + str(len(df_train)) + " test : " + str(
                len(df_test)))
            folds.append({'train_interactions': copy.copy(df_train), 'test_interactions': copy.copy(df_test)})

        return folds

    def stratified_folds(self, no_of_folds=10, stratified_by=constants.Student_stratified, limi_student=1000):
        folds = []
        sfolds = []
        fold = None
        df_data = self.interactions()
        if stratified_by == constants.Student_stratified:
            student_list = df_data[constants.student_field].unique()
            split_student_fold_count = int(np.ceil(len(student_list) * (1 / no_of_folds)))
            for i in range(no_of_folds):
                sfolds[i] = student_list[
                    0 + (i * split_student_fold_count), split_student_fold_count + (i * split_student_fold_count)]

        df_train = None
        df_test = None

        for i in range(no_of_folds):
            df_train = None
            df_test = None
            train_student = [

            ]
            test_student = sfolds[i]
            df_train = df_data[(df_data[constants.student_field].isin(train_student))][constants.interactionid_field]
            df_test = df_data[df_data[constants.student_field] == None][constants.interactionid_field]
            for student in test_student:
                df_temp_student = df_data[(df_data[constants.student_field] == student)]
                split_place = int(np.ceil(split_by_interaction * len(df_temp_student)))
                df_train = df_train.append(df_temp_student[0:split_place][constants.interactionid_field])
                df_test = df_test.append(df_temp_student[split_place + 1:][constants.interactionid_field])
            # print(student_list)
            print("Fold No: " + str(i) + " no of interactions - train  : " + str(len(df_train)) + " test : " + str(
                len(df_test)))
            folds.append({'train_interactions': copy.copy(df_train), 'test_interactions': copy.copy(df_test)})

        return folds

    def get_interaction_concept_attempts(self, dconcepts):

        interaction_concept_attempts = {}
        studentwise_kcwise_current_attempts = {}
        studentwise_kcwise_current_attempts_success = {}
        column_list = []
        # for concept in dconcepts.concept_list:
        #     column_list.append(constants.kc_attempt_prefix + constants)
        #     column_list.append(constants.kc_in_step_prefix + constants)
        #     column_list.append(constants.kc_success_prefix + constants)
        #     column_list.append(constants.kc_failure_prefix + constants)

        df_concept_interaction_details = pd.DataFrame(columns=column_list)
        ## Initialize studentwise_kcwise_current_performance
        for student in self.interactions()[constants.student_field].unique():
            studentwise_kcwise_current_attempts[student] = {concept: 0 for concept in dconcepts.concept_list}
            studentwise_kcwise_current_attempts_success[student] = {concept: 0 for concept in dconcepts.concept_list}


        ## Assign Previous Attempts
        ## Assign Previous Success
        for index, row in self.interactions().iterrows():
            itemid = row[constants.item_field]

            interaction_concept_attempts[row[constants.interactionid_field]] = {
                concept: (studentwise_kcwise_current_attempts[row[constants.student_field]][concept],
                          studentwise_kcwise_current_attempts_success[row[constants.student_field]][concept])
                for (concept, weight) in dconcepts.item2concept_dict[itemid]}


            for (concept, weight) in dconcepts.item2concept_dict[itemid]:
                studentwise_kcwise_current_attempts[row[constants.student_field]][concept] += 1
            if row[constants.performance_field] > 0:
                for (concept, weight) in dconcepts.item2concept_dict[itemid]:
                    studentwise_kcwise_current_attempts_success[row[constants.student_field]][concept] += 1

        return interaction_concept_attempts

    def get_interaction_concept_attempts_activities(self, dconcepts):

        interaction_concept_attempts = {}
        studentwise_kcwise_current_attempts = {}
        studentwise_kcwise_current_attempts_success = {}
        studentwise_kcwise_current_read_attempts = {}
        studentwise_kcwise_current_skip_attempts = {}
        column_list = []
        # for concept in dconcepts.concept_list:
        #     column_list.append(constants.kc_attempt_prefix + constants)
        #     column_list.append(constants.kc_in_step_prefix + constants)
        #     column_list.append(constants.kc_success_prefix + constants)
        #     column_list.append(constants.kc_failure_prefix + constants)

        df_concept_interaction_details = pd.DataFrame(columns=column_list)
        ## Initialize studentwise_kcwise_current_performance

        for student in self.interactions()[constants.student_field].unique():
            studentwise_kcwise_current_attempts[student] = {concept: 0 for concept in dconcepts.concept_list}
            studentwise_kcwise_current_attempts_success[student] = {concept: 0 for concept in dconcepts.concept_list}
            studentwise_kcwise_current_read_attempts[student] = {concept: 0 for concept in dconcepts.concept_list}
            studentwise_kcwise_current_skip_attempts[student] = {concept: 0 for concept in dconcepts.concept_list}

        ## Assign Previous Attempts
        ## Assign Previous Success
        for index, row in self.interactions().iterrows():

            itemid = row[constants.item_field]

            interaction_concept_attempts[row[constants.interactionid_field]] = {
                concept: (studentwise_kcwise_current_attempts[row[constants.student_field]][concept],
                          studentwise_kcwise_current_attempts_success[row[constants.student_field]][concept],
                          studentwise_kcwise_current_read_attempts[row[constants.student_field]][concept],
                          studentwise_kcwise_current_skip_attempts[row[constants.student_field]][concept])

                for (concept, weight) in dconcepts.item2concept_dict[itemid]}

            if row[constants.interaction_type] == constants.interaction_type_quiz:
                for (concept, weight) in dconcepts.item2concept_dict[itemid]:
                    studentwise_kcwise_current_attempts[row[constants.student_field]][concept] += 1
                if row[constants.performance_field] > 0:
                    for (concept, weight) in dconcepts.item2concept_dict[itemid]:
                        studentwise_kcwise_current_attempts_success[row[constants.student_field]][concept] += 1
            else:
                if row[constants.read_behaviour_field] == 1:
                    for (concept, weight) in dconcepts.item2concept_dict[itemid]:
                        studentwise_kcwise_current_skip_attempts[row[constants.student_field]][concept] += 1
                else:
                    for (concept, weight) in dconcepts.item2concept_dict[itemid]:
                        studentwise_kcwise_current_read_attempts[row[constants.student_field]][concept] += 1

        return interaction_concept_attempts

    def get_factor_analysis_interaction_folds(self, folds, fields=[]):

        if len(fields) == 0:
            fields = self.interactions().columns

        fa_folds = {}
        for fold in folds:
            foldid = len(fa_folds)
            fa_folds[foldid] = {}
            for fold_type in fold:
                # print(foldid, fold_type, len(fold[fold_type]))
                df_temp_interactions = \
                self.interactions()[self.interactions()[constants.interactionid_field].isin(fold[fold_type].unique())][
                    fields]
                fa_folds[foldid][fold_type] = df_temp_interactions
        return fa_folds

    def get_factor_analysis_interaction_folds_activities(self, folds, fields=[]):

        if len(fields) == 0:
            fields = self.interactions().columns

        fa_folds = {}
        for fold in folds:
            foldid = len(fa_folds)
            fa_folds[foldid] = {}
            for fold_type in fold:
                # print(foldid, fold_type, len(fold[fold_type]))
                df_temp_interactions = \
                    self.interactions()[
                        self.interactions()[constants.interactionid_field].isin(fold[fold_type].unique())][
                        fields]
                fa_folds[foldid][fold_type] = df_temp_interactions
        return fa_folds

    def get_factor_analysis_interaction_folds_type(self, folds, fields=[]):

        if len(fields) == 0:
            fields = self.interactions().columns

        fa_folds = {}
        for fold in folds:
            foldid = len(fa_folds)
            fa_folds[foldid] = {}
            for fold_type in fold:
                # print(foldid, fold_type, len(fold[fold_type]))
                df_temp_interactions = \
                    self.interactions()[
                        self.interactions()[constants.interactionid_field].isin(fold[fold_type].unique())][
                        fields]
                fa_folds[foldid][fold_type] = df_temp_interactions
        return fa_folds

    def get_concept_distribution(self, dconcepts):

        # itemlist = self.interactions()[constants.item_field].unique()
        # print(" ItemName ", end=" ")
        # for item in itemlist:
        #     if item.startswith("q"):
        #         print(item, end=" ")
        # print("")
        # for item1 in itemlist:
        #     if  not item1.startswith("q"):
        #         print(item1, end=" ")
        #     else:
        #         continue
        #     l_kc1 = [concept for (concept, weight) in dconcepts.item2concept_dict[item1]]
        #     for item2 in itemlist:
        #         if item2.startswith("q"):
        #             l_kc2 = [concept for (concept, weight) in dconcepts.item2concept_dict[item2]]
        #             print(str(len(intersect(l_kc1,l_kc2)))+"_"+"_".join(intersect(l_kc1,l_kc2)), end=" ")
        #     print("")
        #
        # print(" ItemName ", end=" ")
        # for item in itemlist:
        #     if item.startswith("q"):
        #         print(item, end=" ")
        # print("")
        # for item1 in itemlist:
        #     if  not item1.startswith("q"):
        #         print(item1, end=" ")
        #     else:
        #         continue
        #     l_kc1 = [concept for (concept, weight) in dconcepts.item2concept_dict[item1]]
        #     for item2 in itemlist:
        #         if item2.startswith("q"):
        #             l_kc2 = [concept for (concept, weight) in dconcepts.item2concept_dict[item2]]
        #             print(len(intersect(l_kc1,l_kc2)), end=" ")
        #     print("")

        interaction_concept_attempts = {}
        studentwise_kcwise_current_attempts = {}
        studentwise_kcwise_current_attempts_success = {}
        column_list = []

        ## Initialize studentwise_kcwise_current_performance
        for student in self.interactions()[constants.student_field].unique():
            studentwise_kcwise_current_attempts[student] = {concept: 0 for concept in dconcepts.concept_list}
            studentwise_kcwise_current_attempts_success[student] = {concept: 0 for concept in dconcepts.concept_list}

        ## Assign Previous Attempts
        ## Assign Previous Success
        for index, row in self.interactions().iterrows():
            itemid = row[constants.item_field]

            interaction_concept_attempts[row[constants.interactionid_field]] = {
                concept: (studentwise_kcwise_current_attempts[row[constants.student_field]][concept],
                          studentwise_kcwise_current_attempts_success[row[constants.student_field]][concept])
                for (concept, weight) in dconcepts.item2concept_dict[itemid]}

            for (concept, weight) in dconcepts.item2concept_dict[itemid]:
                studentwise_kcwise_current_attempts[row[constants.student_field]][concept] += 1
            if row[constants.performance_field] > 0:
                for (concept, weight) in dconcepts.item2concept_dict[itemid]:
                    studentwise_kcwise_current_attempts_success[row[constants.student_field]][concept] += 1

        return interaction_concept_attempts


## should I add rank in the concept dictionary -- I am not very sure though -- keep it for later work

class Concepts:

    def __init__(self, name, concept_file, weighted=False, topn=-1, dummy=True):
        df_concept = pd.read_csv(open(concept_file), header=0)
        df_concept[constants.item_field] = df_concept[constants.item_field].astype(str)
        self.weighted = weighted
        self.topn = topn
        self.conceptlistname = name
        self.concept_list = []
        self.item2concept_dict = {}
        self.concept2item_dict = {}
        self.fillConcept_dict(df_concept)

    def fillConcept_dict(self, df_concept):
        df_item_group = df_concept.groupby([constants.item_field]).mean()

        for index, row in df_item_group.iterrows():
            itemid = str(index)
            self.item2concept_dict[itemid] = []
            df_item_concept = df_concept[df_concept[constants.item_field] == itemid]
            df_item_concept.sort_values(by=['weight'])
            self.item2concept_dict[itemid] = [(row['concept'], round(float(row['weight']), 2)) for index, row in
                                              df_item_concept.head(self.topn).iterrows()]

            for (concept, weight) in self.item2concept_dict[itemid]:
                concept = str(concept)
                if concept not in self.concept_list:
                    self.concept_list.append(concept)

                if concept not in self.concept2item_dict:
                    self.concept2item_dict[concept] = []

                if itemid not in self.concept2item_dict[concept]:
                    self.concept2item_dict[concept].append(itemid)

