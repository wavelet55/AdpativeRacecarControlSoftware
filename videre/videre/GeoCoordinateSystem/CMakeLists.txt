project(GeoCoordinateSystem)

message("[videre] Setting up Geo-Coordinate System OBJECT library build")

set(SRCS_H 
        ${CMAKE_CURRENT_SOURCE_DIR}/AzimuthElevation_t.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GeoCoordinateSystem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/LatLonAltStruct.h
        ${CMAKE_CURRENT_SOURCE_DIR}/LatLonUtmTransformations.h
        )

set(SRCS_CPP
        ${CMAKE_CURRENT_SOURCE_DIR}/AzimuthElevation_t.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GeoCoordinateSystem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/LatLonAltStruct.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/LatLonUtmTransformations.cpp
        )


add_library(${PROJECT_NAME} STATIC ${SRCS_CPP} ${SRCS_H})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE rabit
        Utils
        ImageProcessingCommonLib)