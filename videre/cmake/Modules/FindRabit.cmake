# - Try to find ZMQ
# Once done this will define
# ZMQ_FOUND - System has ZMQ
# ZMQ_INCLUDE_DIRS - The ZMQ include directories
# ZMQ_LIBRARIES - The libraries needed to use ZMQ
# ZMQ_DEFINITIONS - Compiler switches required for using ZMQ

find_path ( RABIT_INCLUDE_DIR rabit/RabitConfig.h )
find_library ( RABIT_LIBRARIES NAMES rabit )

set ( RABIT_LIBRARY ${RABIT_LIBRARIES} )
set ( RABIT_INCLUDE_DIR ${RABIT_INCLUDE_DIR} )

include ( FindPackageHandleStandardArgs )
# handle the QUIETLY and REQUIRED arguments and set ZMQ_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args ( Rabit DEFAULT_MSG RABIT_LIBRARIES RABIT_INCLUDE_DIR )
