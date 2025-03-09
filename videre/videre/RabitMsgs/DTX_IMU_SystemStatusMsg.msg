#This Message is for the status of the DTX Control Board.
uint64  timestamp	# time since system start (microseconds)
uint8   gf360_cmd_state	# State the GF360 is in
uint8   switch_states	# State of all input switches
uint8   exercise_init_state   #GF360DTXExerciseInitStates_e
uint8   exercise_user_situate_state   #GF360DTXExerciseInitStates_e
bool    main_air_solenoid_state
bool    main_cylender_port_a_state
bool    main_cylender_port_b_state
float32 main_resistance_regulator_pressure    #[0, 100.0] percent
float32 counter_ballance_regulator_pressure   #[0, 100.0] percent
float32 spare_analog_input		#[0, 10.0] volts
float32 left_motor_position		#[0, 120.0] degrees
float32 right_motor_position		#[0, 120.0] degrees
float32 harness_updwn_position		#[0, 100.0] percent
float32 harness_left_right_position	#[-100.0, 100.0] percent
float32 harness_rotation_position	#[-160.0, 160.0] degrees
