# model
PFA = 'pfa'
LFA = 'lfa'

# fields
student_field = 'student_id'
item_field = 'item_id'
concept_field = 'concept'
starttime_field = 'starttime'
endtime_field = 'endtime'
group_field = 'group_id'
interactionid_field = 'interaction_id'
interaction_type = 'interaction_type'
performance_field = 'is_interaction_success'


conceptwise_attemp_field = 'kc_attempt'
kc_attempt_prefix = "KC#attempts_"
kc_success_prefix = "KC#success_"
kc_failure_prefix = "KC#failure_"
kc_in_step_prefix = "KC#instep_"
kc_read_attempt_prefix = "KC#readattempt_"
step_hardness_prefix = "STEP#hardness_"
student_prior_prefix = "STU#Prior_"
outcome_prefix = "OUT#come"

#To Be removed constants in future
questionid_field = "que_id"
pageid_field = "page"
debug = False

#Split Details
split_by_student = 0.5
fields = [interactionid_field, interaction_type, student_field, item_field, performance_field]

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
