# sm_learner_analysis
PFA, LFA and other frameworks for student modeling (ongoing)

Code Assumes there are atleast 2 students not a single student

# 10 Folds creation
First students are divided in two parts - based on contants.split_by_student
first half - STU_FL with complete data used for training 
second half - STU_SL again divided in two parts - based on contants.split_by_interaction
STU_SL first half is added to training and second to testing
