uint8     command                   # Ref: GF360_RecordControlCmd_e
int32     file_index                # Record File Index Number
uint8[36] file_guid                 # Guid / UUID to be included in the file header.
uint32    record_version            # Over-all Version number for Records.
uint8[36] exercise_guid             # Guid / UUID associated with an exercise.
uint8     exercise_status           # Status for the current Exercise
