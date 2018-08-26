import copy
import time

from sklearn.linear_model import LogisticRegression

from experiments.config import constants
from lib import index

start_time = time.time()
import pickle


class PFA:

    def __init__(self, concept_list, penalty='l2', C=0.01, max_iter=1000, name="expName"):

        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.folds = []
        self.name = name
        self.concept_list = concept_list
        self.concept2inx, self.inx2concept = index.getData2Index(self.concept_list)

        self.pfa_columns = self.def_pfa_columns()

    def def_pfa_columns(self):
        pfa_columns = []
        pfa_columns.append(constants.item_field)
        for inx in range(len(self.concept_list)):
            concept_inx = str(inx)
            pfa_columns.append(constants.kc_in_step_prefix + concept_inx)
            pfa_columns.append(constants.kc_success_prefix + concept_inx)
            pfa_columns.append(constants.kc_failure_prefix + concept_inx)
        return pfa_columns

    def loadClass(self, objfile):
        self = pickle.load(open(objfile))

    def setFolds(self, i_interactions_folds, interaction_concept_details):

        for fold_id in i_interactions_folds:
            if constants.train_fold in i_interactions_folds[fold_id]:


    def fitFolds(self):
        try:
            if len(self.folds) == 0:
                raise ValueError("No folds set!")

            for fold in self.folds:
                if constants.model_fold in fold:
                    print("already trained")
                    continue

                X_train = fold[constants.train_fold]
                y_train = fold[constants.test_fold]
                clf_l2_LR = LogisticRegression(C=self.C, penalty=self.penalty, max_iter=self.max_iter)
                clf_l2_LR.fit(X_train, y_train)
                self.fold[constants.model_fold] = copy.copy(clf_l2_LR)
        except ValueError as ve:
            print(ve)

    def saveExperiment(self, directory):
        pickle.dump(self, open(directory + "/" + self.name, 'wb'))

    def predictFolds(self):

        try:
            if len(self.folds) == 0:
                raise ValueError("No folds set!")

            for fold in self.folds:
                if 'model' not in fold:
                    print("fold not trained")
                    self.fitFolds()

                X_test = fold[constants.test_fold]
                fold[constants.prediction_fold] = fold[constants.model_fold].predict(X_test)

        except ValueError as ve:
            print(ve)
        except:
            print("Errors with prediction")

# def makeFolds(student_interaction_file, kc_file, top_kc=-1, no_of_folds=10):
#     model_folder = model_folder + "/{i}/"
#     model_folder = model_folder.replace("{i}", str(fold))
#
#     doc_to_term_matrix = data_folder + "/" + model_folder + 'MapDocToTerms.csv'
#     doc_to_qmatrix_id = data_folder + "/" + model_folder + 'MapDocIdToQmatrixID.csv'
#     kc_to_qmatrix_id = data_folder + "/" + model_folder + 'MapTermNameToQmatrixID.csv'
#     test_file = data_folder + "/" + model_folder + "Test0.csv"
#     train_file = data_folder + "/" + model_folder + "Train0.csv"
#
#     df_train = pd.read_csv(train_file, header=0)
#     df_test = pd.read_csv(test_file, header=0)
#     doc_kc_weight = {}
#     d2Q_dict = {}
#     Q2d_dict = {}
#     T2Q_dict = {}
#     d2T_dict = {}
#     Q2T_dict = {}
#     with open(weight_file_doci, mode='r') as infile:
#         reader = csv.reader(infile)
#         headers = next(reader, None)
#         for rows in reader:
#             id = str(rows[0])
#             if id not in doc_kc_weight:
#                 doc_kc_weight[id] = {}
#             if weights:
#                 doc_kc_weight[id][rows[1]] = float(rows[2])
#             else:
#                 doc_kc_weight[id][rows[1]] = 1
#
#     with open(weight_file_questioni, mode='r') as infile:
#         reader = csv.reader(infile)
#         headers = next(reader, None)
#         for rows in reader:
#             id = "q" + str(rows[0])
#             if id not in doc_kc_weight:
#                 doc_kc_weight[id] = {}
#             if weights:
#                 doc_kc_weight[id][rows[1]] = float(rows[2])
#             else:
#                 doc_kc_weight[id][rows[1]] = 1
#
#     with open(doc_to_qmatrix_id, mode='r') as infile:
#         reader = csv.reader(infile)
#         for rows in reader:
#             k = rows[0]
#             v = int(rows[1])
#             d2Q_dict[k] = v
#             Q2d_dict[v] = k
#
#     with open(kc_to_qmatrix_id, mode='r') as infile:
#         reader = csv.reader(infile)
#         for rows in reader:
#             # print(rows)
#             k = rows[0]
#             v = int(rows[1])
#             T2Q_dict[k] = v
#             Q2T_dict[v] = k
#
#     k_i = 0
#     with open(doc_to_term_matrix, mode='r') as infile:
#         reader = csv.reader(infile)
#         for rows in reader:
#             k = Q2d_dict[k_i]
#             v = [Q2T_dict[j] for j in range(len(rows)) if float(rows[j]) > 0]
#             k_i += 1
#             d2T_dict[k] = v
#
#     student_dict = {}
#     i = 0
#     with open(train_file, mode='r') as infile:
#         reader = csv.reader(infile)
#         for rows in reader:
#             k = rows[0]
#             v = i
#             if k not in student_dict:
#                 student_dict[k] = v
#                 i += 1
#
#     no_of_unique_kcs = len(T2Q_dict)
#     no_of_unique_students = len(student_dict)
#
#     pfa_column_dict = {}
#     for k in T2Q_dict:
#         pfa_column_dict[k + "_wrong"] = 0
#         pfa_column_dict[k + "_correct"] = 0
#         pfa_column_dict[k + "_beta"] = 0
#         pfa_column_dict['q' + k + "_wrong"] = 0
#         pfa_column_dict['q' + k + "_correct"] = 0
#         pfa_column_dict['q' + k + "_beta"] = 0
#
#     dropindex = []
#     if not 'All' in Train_on:
#         for index, row in df_train.iterrows():
#             drop = True
#             if 'Read' in Train_on and not str(row[1]).startswith('q'):
#                 drop = False
#             if 'Quiz' in Train_on and str(row[1]).startswith('q'):
#                 drop = False
#             if drop:
#                 dropindex.append(index)
#         df_train = df_train.drop(dropindex)
#
#     dropindex = []
#     if not 'All' in Test_on:
#         for index, row in df_test.iterrows():
#             drop = True
#             if 'Read' in Test_on and not str(row[0]).startswith('q'):
#                 drop = False
#             if 'Quiz' in Test_on and str(row[0]).startswith('q'):
#                 drop = False
#             if drop:
#                 dropindex.append(index)
#         df_test = df_test.drop(dropindex)
#
#     listrequired = []
#     # put the original column names in a python list
#     X_train, y_train, stepwise_current_kc_count = generateTrainFileWeighted_bpm(df_train, pfa_column_dict,
#                                                                                 doc_kc_weight)
#     X_test, y_test = generateTestFileWeighted_bpm(df_test, stepwise_current_kc_count, pfa_column_dict, d2T_dict,
#                                                   doc_kc_weight)
#
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#     # Static
#     #     print("-- fold",fold)
#
#     fold = {'xtrain': copy.copy(X_train), 'ytrain': copy.copy(y_train), 'xtest': copy.copy(X_test)}
#
#     return fold
