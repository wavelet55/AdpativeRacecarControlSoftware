#!/bin/bash

# To use this script you must have Google's protoc builder installed
# (https://code.google.com/p/protobuf/). 

echo Building Google protocol buffers for cpp and python...
echo ""

MESSAGESPATH=messages
SOURCE=gen_source
MESSAGES=$MESSAGESPATH/*.proto
CPPPATHOUT=$SOURCE/cpp
PYTHONPATHOUT=$SOURCE/python
#Videre Protobuf message path:
VIDEREPBMSGPATH=../videre/videre/ProtobufMessages


# Since it's easiest, generate c++ and python protocs first
for proto in $MESSAGES
do
	echo [+] Building cpp protos for $proto
	protoc -I=$MESSAGESPATH --cpp_out=$CPPPATHOUT  $proto
	echo [+] Building python protos for $proto
	protoc -I=$MESSAGESPATH --python_out=$PYTHONPATHOUT  $proto
done

cp -f $CPPPATHOUT/vision_messages.pb.cc $VIDEREPBMSGPATH/vision_messages.pb.cpp
cp -f $CPPPATHOUT/vision_messages.pb.h $VIDEREPBMSGPATH/vision_messages.pb.h

