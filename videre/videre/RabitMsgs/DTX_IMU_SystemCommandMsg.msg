#This Message is for the control of the DTX Control Board.
#The gf360_cmd_state is the requested GF360 operational state.
#It may be used for manually controlling some of the GF360 system.
uint64  timestamp	# time since system start (microseconds)
uint8   gf360_cmd_state	# State the GF360 should be in or go to
bool    main_air_solenoid_enable
bool    main_cylinder_port_a_enable
bool    main_cylinder_port_b_enable
float32 main_resistance_regultor_ctrl   #[0, 100.0] percent
float32 counter_ballance_regultor_ctrl   #[0, 100.0] percent
float32 left_motor_position		#[0, 120.0] degrees
float32 right_motor_position		#[0, 120.0] degrees
float32 harness_updwn_motor_position	#[0, 100.0] percent
