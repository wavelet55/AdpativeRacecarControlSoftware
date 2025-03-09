project(ProtobufMessages)

message("[videre] Setting up ProtobufMessages System OBJECT library build")

set(SRCS_H 
        ${CMAKE_CURRENT_SOURCE_DIR}/vision_messages.pb.h
        )

set(SRCS_CPP
        ${CMAKE_CURRENT_SOURCE_DIR}/vision_messages.pb.cpp
        )


add_library(${PROJECT_NAME} STATIC ${SRCS_CPP} ${SRCS_H})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(${PROJECT_NAME} PRIVATE rabit Utils ImageProcessingCommonLib)