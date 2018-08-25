# model
PFA = 'pfa'
LFA = 'lfa'

# fields
student_field = 'student_id'
item_field = 'item_id'
concept_field = 'concept'
starttime_field = 'starttime'
endtime_field = 'endtime'
interactionid_field = 'interaction_id'
interaction_type = 'interaction_type'
performance_field = 'is_correct'
conceptwise_attemp_field = 'kc_attempt'
concept_field_prefix = "CS#_"

#To Be removed constants in future
questionid_field = "que_id"
pageid_field = "page"
debug = True

#Split Details
split_by_student = 0.5
fields = [interactionid_field, interaction_type, student_field, item_field, performance_field]

# constants_interaction_folds
train_fold = "train_interactions"
test_fold = "test_interactions"
model_fold = "model"
