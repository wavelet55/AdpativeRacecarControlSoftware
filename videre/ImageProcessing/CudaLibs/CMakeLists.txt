project(CudaImageProcessingLibs)

#setup the NVidia NVCC toolchain
# [macro within cmake/utils.cmake]
cuda_nvcc_options_setup()

find_package (Boost REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(CUDA QUIET REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(Log4Cxx REQUIRED)
include_directories(${LOG4CXX_INCLUDE_DIR})

add_definitions(-DCUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

set(SRCS_H
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/Math/SmVectors.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/Math/SmVectors_Cuda.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/Utils/exception_cuda.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/Utils/helper_cuda.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/Utils/helper_cuda_math.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/Utils/helper_cuda_string.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/Utils/helper_cuda_timer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/Utils/PixelColorValueCuda.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/CudaImageKeypointsAndFeatures.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/ImageColorThresholdKernel.h
        ${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobDetectorOpenCVSimple/CudaBlobTargetDetectorOpenCVSimple.h
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/Blob.h
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/BlobExtraction.h
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/BlobLibraryConfiguration.h
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/BlobResult.h
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/CudaBlobTargetDetector.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GlyphDetector/Glyph.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GlyphDetector/GlyphModel.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GlyphDetector/DetectGlyphs.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GlyphDetector/HeadPosition.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GlyphDetector/TrackHeadProcess.h
        )

set(SRCS_CPP
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/Math/SmVectors.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GlyphDetector/Glyph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GlyphDetector/GlyphModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GlyphDetector/DetectGlyphs.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GlyphDetector/HeadPosition.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GlyphDetector/TrackHeadProcess.cpp
       )

set(SRCS_CU
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/Math/SmVectors_Cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/Utils/PixelColorValueCuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/CudaImageKeypointsAndFeatures.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaCommon/ImageColorThresholdKernel.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobDetectorOpenCVSimple/CudaBlobTargetDetectorOpenCVSimple.cu
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/Blob.cu
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/BlobExtraction.cu
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/BlobResult.cu
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/CudaBlobTargetDetector.cu
        )


cuda_add_library(${PROJECT_NAME} SHARED ${SRCS_CPP} ${SRCS_CU} ${SRCS_H})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME}
        ${Boost_LIBRARIES}
        ${ARMADILLO_LIBRARIES}
        ${OpenCV_LIBS}
        ${LOG4CXX_LIBRARIES}
        ImageProcessingCommonLib
        )

