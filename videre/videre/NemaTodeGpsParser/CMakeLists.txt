project(NemaTodeGpsParser)

message("[videre] Setting up NemaTodeGpsParser System OBJECT library build")

set(SRCS_H 
        ${CMAKE_CURRENT_SOURCE_DIR}/Event.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GPSFix.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GPSService.h
        ${CMAKE_CURRENT_SOURCE_DIR}/nmea.h
        ${CMAKE_CURRENT_SOURCE_DIR}/NMEACommand.h
        ${CMAKE_CURRENT_SOURCE_DIR}/NMEAParser.h
        ${CMAKE_CURRENT_SOURCE_DIR}/NumberConversion.h
        )

set(SRCS_CPP
        ${CMAKE_CURRENT_SOURCE_DIR}/GPSFix.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GPSService.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/NMEACommand.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/NMEAParser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/NumberConversion.cpp
        )


add_library(${PROJECT_NAME} STATIC ${SRCS_CPP} ${SRCS_H})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(${PROJECT_NAME} PRIVATE )