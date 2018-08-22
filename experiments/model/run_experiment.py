import os
import time

import pandas as pd

from preprocessor import DataProcessor, Concept

start_time = time.time()

# def foldGeneration(data_folder,model_folder,fold,weight_file_doci,weight_file_questioni,weights=False,Train_on=['All'],Test_on=['All'],sep_param=True ):
#
#     model_folder = model_folder + "/{i}/"
#     model_folder = model_folder.replace("{i}",str(fold))
#
#     doc_to_term_matrix = data_folder+"/"+model_folder+'MapDocToTerms.csv'
#     doc_to_qmatrix_id = data_folder+"/"+model_folder+'MapDocIdToQmatrixID.csv'
#     kc_to_qmatrix_id = data_folder+"/"+model_folder+'MapTermNameToQmatrixID.csv'
#     test_file = data_folder+"/"+model_folder+"Test0.csv"
#     train_file = data_folder+"/"+model_folder+"Train0.csv"
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
#             id = "q"+str(rows[0])
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
#         pfa_column_dict[k  + "_wrong"] = 0
#         pfa_column_dict[k  + "_correct"] = 0
#         pfa_column_dict[k + "_beta"] = 0
#         pfa_column_dict['q'+ k + "_wrong"] = 0
#         pfa_column_dict['q'+ k + "_correct"] = 0
#         pfa_column_dict['q'+ k + "_beta"] = 0
#
#     dropindex = []
#     if not 'All' in Train_on:
#         for index,row in df_train.iterrows():
#             drop = True
#             if 'Read' in Train_on and not str(row[1]).startswith('q'):
#                 drop=False
#             if 'Quiz' in Train_on and str(row[1]).startswith('q'):
#                 drop = False
#             if drop:
#                 dropindex.append(index)
#         df_train = df_train.drop(dropindex)
#
#
#     dropindex = []
#     if not 'All' in Test_on:
#         for index,row in df_test.iterrows():
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
#     X_train,y_train,stepwise_current_kc_count = generateTrainFileWeighted_bpm(df_train,pfa_column_dict,doc_kc_weight)
#     X_test,y_test = generateTestFileWeighted_bpm(df_test,stepwise_current_kc_count,pfa_column_dict,d2T_dict,doc_kc_weight)
#
#     scaler =  StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
# # Static
# #     print("-- fold",fold)
#
#     fold = {'xtrain':copy.copy(X_train),'ytrain':copy.copy(y_train),'xtest':copy.copy(X_test)}
#
#
#     return fold


if __name__ == '__main__':
    data_folder = "data/"
    Discretization_type = 'DiscPerDoc'
    concept_level_field = 'itemid'
    interaction_data_file = 'data/interaction_data/data_{dataname}.pkl'
    student_split_percent = .5
    interactions_split_percent = .5
    student_interaction_file = "Fall2017_reading_quiz_logs.pageaction_ignored_nomergethreshold.csv"
    concept_file_folder = 'concept_files/'

    df_interactions = pd.read_csv(data_folder + student_interaction_file)
    df_interactions[concept_level_field] = ""
    for index, row in df_interactions.iterrows():
        if row['is_question'] == 1:
            df_interactions.loc[index, concept_level_field] = str(df_interactions.loc[index]['question_id'])
        else:
            df_interactions.loc[index, concept_level_field] = str(df_interactions.loc[index]['page'])

    int_data = DataProcessor(df_interactions)
    concept_dictionary = {}
    interaction_folds = int_data.studentwise_interaction_folds(no_of_folds=10, split_by_student=0.5,
                                                               split_by_interaction=.5)
    for concept_file in os.listdir(data_folder + concept_file_folder):
        concept_obj = Concept(name=concept_file.split(".")[0],
                              concept_file=data_folder + concept_file_folder + concept_file, weighted=True, topn=5)

    print("end")

    # data = []
    # result = []
    # result_dynamic = []
    # fwrite = open("output.all.txt.all.read.non_weighted.top.kcprior",'w+')
    # fwrite.write("==== New output ==")
    # base_dir_list = [
    #             'Log_From_DB_16_Spring/Experiments/IR_kb/DiscPerDoc/BookAll_ConceptTFIDFTop5/','Log_From_DB_16_Spring/Experiments/IR_kb/DiscPerDoc/BookAll_ConceptTFIDFTop10/']
    # #
    # # 'Log_From_DB_16_Spring/Experiments/IR_kb/DiscPerDoc/BookAll_ConceptTFIDFTop15/',
    # # 'Log_From_DB_16_Spring/Experiments/IR_kb/DiscPerDoc/BookAll_ConceptTFIDFTop20/',
    # # 'Log_From_DB_16_Spring/Experiments/IR_kb/DiscPerDoc/BookAll_ConceptTFIDFTop25/',
    # # 'Log_From_DB_16_Spring/Experiments/IR_kb/DiscPerDoc/BookAll_ConceptTFIDFTop30/'
    # #['All','Read'],['All','Quiz'],['Read','Quiz'],
    #
    # for experiments in [['Quiz','Read']]:
    #     exp_train_on = experiments[0]
    #     exp_test_on = experiments[1]
    #     for base_dir in base_dir_list:
    #         weight_file_doc = 'Textbook/DocVector/Concept/{exp}/{exp_t}.16spring.irbook.csv'
    #         weight_file_quiz = 'Textbook/DocVector/Concept/{exp}/{exp_t}.16spring.irquestions.csv'
    #
    #         models= ['Read+Quiz']
    #         for model_dir in models:
    #             for pathi in os.listdir(base_dir):
    #                 if pathi.startswith("16Fall") and '5' in pathi :
    #                     exp=pathi.replace("16Fall","").replace('0.5',"").split(".")[0]
    #
    #
    #                     if 'LDA' in pathi:
    #                         print("nothing for now")
    #                         # exp = pathi.replace("16Fall", "").split("_")[0]
    #                         # topics = exp +"."+pathi.replace("16Fall", "").split("_")[1]
    #                         # input_weight_file = weight_file_doc
    #                         # input_weight_file_doc = input_weight_file.replace("{exp}",exp).replace("{exp_t}",topics)
    #                         # input_weight_file = weight_file_quiz
    #                         # input_weight_file_quiz = input_weight_file.replace("{exp}",exp).replace("{exp_t}",topics)
    #                         continue
    #                     else:
    #                         input_weight_file = weight_file_doc
    #                         input_weight_file_doc = input_weight_file.replace("{exp}",exp).replace("{exp_t}",exp)
    #                         input_weight_file = weight_file_quiz
    #                         input_weight_file_quiz = input_weight_file.replace("{exp}",exp).replace("{exp_t}",exp)
    #
    #                     path = base_dir+"/"+pathi
    #
    #                     result = []
    #                     folds = []
    #                     for i in range(10):
    #                          f = foldGeneration(path,model_dir,i,input_weight_file_doc, input_weight_file_quiz,weights=False,Train_on=[exp_train_on], Test_on=[exp_test_on])
    #                          folds.append(f)
    #
    #
    #                         result.append([
    #                         roc_auc_score(y_fold_true, y_fold_predictions),
    #                         mean_squared_error(y_fold_true, y_fold_predictions),
    #                         ])
    #                     print(base_dir, pathi, exp, exp_train_on, exp_test_on,np.mean(result,axis=0))
    #                     # print()
    #                     # print(np.mean(result_dynamic,axis=0))
    #                     print("--- %s minutes ---" % (time.time() - start_time))
    #                     fwrite.write(path+"\n")
    #                     fwrite.write(model_dir+"\n")
    #                     fwrite.write(str(exp_train_on)+":"+str(exp_test_on))
    #                     fwrite.write(' '.join(list(map(str,np.mean(result,axis=0)))))
    #                     fwrite.write("\n")
