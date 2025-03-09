# - Try to find SocketCan
# Once done this will define
# SocketCan_FOUND - System has SocketCan
# SocketCan_INCLUDE_DIRS - The SocketCan include directories
# SocketCan_LIBRARIES - The libraries needed to use SocketCan
# SocketCan_DEFINITIONS - Compiler switches required for using SocketCan

find_path ( SOCKETCAN_INCLUDE_DIR libsocketcan.h )
find_library ( SOCKETCAN_LIBRARIES NAMES socketcan )

set ( SOCKETCAN_LIBRARY ${SOCKETCAN_LIBRARIES} )
set ( SOCKETCAN_INCLUDE_DIR ${SOCKETCAN_INCLUDE_DIR} )

include ( FindPackageHandleStandardArgs )
# handle the QUIETLY and REQUIRED arguments and set ZMQ_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args ( SocketCan DEFAULT_MSG SOCKETCAN_LIBRARIES SOCKETCAN_INCLUDE_DIR )
