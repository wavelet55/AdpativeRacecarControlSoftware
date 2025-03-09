
project(ImageProcessingCommonLib)

message("[videre] Building Image Processing Common OBJECT Library")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

find_package (Boost REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(Log4Cxx REQUIRED)
include_directories(${LOG4CXX_INCLUDE_DIR})

set(SRCS_H 
        ${CMAKE_CURRENT_SOURCE_DIR}/BlobDetectorFixedParameters.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ButterworthIIRLPF.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraCalibrationData.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CommonImageProcTypesDefs.h
        ${CMAKE_CURRENT_SOURCE_DIR}/OpenCVMatUtils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/PixelColorValue.h
        ${CMAKE_CURRENT_SOURCE_DIR}/MathUtils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Quaternion.h
        ${CMAKE_CURRENT_SOURCE_DIR}/RollPitchYaw_t.h
        ${CMAKE_CURRENT_SOURCE_DIR}/XYZCoord_t.h
        )

set(SRCS_CPP
        ${CMAKE_CURRENT_SOURCE_DIR}/ButterworthIIRLPF.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraCalibrationData.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CommonImageProcTypesDefs.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/OpenCVMatUtils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/PixelColorValue.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/MathUtils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Quaternion.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/RollPitchYaw_t.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/XYZCoord_t.cpp
        )


add_library(${PROJECT_NAME} STATIC ${SRCS_CPP} ${SRCS_H})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE 
        ${Boost_LIBRARIES}
        ${ARMADILLO_LIBRARIES}
        ${OpenCV_LIBS}
        ${LOG4CXX_LIBRARIES}
        )
