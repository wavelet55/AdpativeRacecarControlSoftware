uint8     record_file_type      # Ref: GF360_RecordStatus_e
uint8     record_file_status    # Ref:
uint8     record_status         # Ref: GF360_RecordStatus_e
int32     file_index            # Record File Index Number
uint8[36] record_guid           # Guid / UUID associated with an exercise or manual record.
uint32    record_index
uint32    data_start_index      #Start index of recorded data
uint32    data_stop_index       #Stop index of recorded data
