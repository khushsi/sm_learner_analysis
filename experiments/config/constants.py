# model
PFA = 'pfa'
LFA = 'lfa'

# fields
student_field = 'student_id'
item_field = 'item_id'
concept_field = 'kc'
starttime_field = 'starttime'
endtime_field = 'endtime'
group_field = 'group_id'
interactionid_field = 'interaction_id'
interaction_type = 'interaction_type'
performance_field = 'is_interaction_success'
read_behaviour_field = 'is_skimming'
interaction_type_quiz = 'Quiz'
interaction_type_read = 'Read'
fields = [interactionid_field, interaction_type, student_field, item_field, performance_field, read_behaviour_field]
# OLI
# concept_field = 'kc'
concept_att_field = 'kc_step_opportunity'
# starttime_field = 'start_time'
# endtime_field = 'end_time'
# performance_field = 'is_interaction_success'
# fields = [interactionid_field, interaction_type, student_field, item_field, performance_field,concept_field,concept_att_field]

conceptwise_attemp_field = 'kc_attempt'
kc_attempt_prefix = "KC#attempts_"
kc_success_prefix = "KC#success_"
kc_failure_prefix = "KC#failure_"
kc_in_step_prefix = "KC#instep_"
kc_read_attempt_prefix = "KC#readattempt_"
kc_skim_attempt_prefix = "KC#skipattempt_"
step_hardness_prefix = "STEP#hardness_"
student_prior_prefix = "STU#Prior_"
outcome_prefix = "OUT#come"

#To Be removed constants in future
questionid_field = "que_id"
pageid_field = "page"
debug = False

# To Be removed constants in future

#Split Details
split_by_student = 0.5


# constants_interaction_folds
train_fold = "train_interactions"
test_fold = "test_interactions"
xtrain = 'xtrain'
ytrain = 'ytrain'
xtest = 'xtest'
ytest = 'ytest'
model_fold = "model"
prediction_fold = "prediction"
vectorizer = 'vec'

#
Student_stratified = 'student_startified'
Item_stratitified = 'item_stratified'
Random_stratified = 'random_stratified'
