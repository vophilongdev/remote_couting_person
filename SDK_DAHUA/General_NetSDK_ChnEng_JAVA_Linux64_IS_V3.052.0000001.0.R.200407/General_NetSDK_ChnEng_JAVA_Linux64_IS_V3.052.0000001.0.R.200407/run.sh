#!/bin/bash
ARG_COUNT=$#
#echo $ARG_COUNT
if [ $ARG_COUNT -lt 2 ]; then
  echo "Usage: run.sh -frame example-name or run.sh -Attendance Attendance-name or run.sh -AutoRegister AutoRegister-name .etc"
  exit 1
fi 

#echo "--> params: "$1 $2
if [ "$1" != "-frame" -a "$1" != "-Attendance" -a "$1" != "-AutoRegister"  -a "$1" != "-FaceRecognition"  -a "$1" != "-Gate"  -a "$1" != "-ThermalCamera" ]; then
  echo "The first param must be -frame or -Attendance or -AutoRegister or -FaceRecognition or -Gate or -ThermalCamera"  
  exit 1
fi 


#linux下是在src下编译运行的，  
cd ./src

#编译   引入需要的依赖的jar以及java文件，以及运行文件
CP+=../libs/jna.jar:
CP+=../libs/examples.jar
#字符支持——此处有误
CP+=:../res/res_en_US.properties
CP+=:../res/res_zh_CN.properties

os=`uname -a`
##指定库的路径    加入动态链接库
if [[ $os =~ "Darwin" ]]; then
	CP+=:../libs/mac64
	echo "--> mac 64 System"
elif [ $(getconf LONG_BIT) = '64' ]; then
    echo "--> linux 64 System."
    export LD_LIBRARY_PATH=../libs/linux64
else
    echo "--> linux 32 System."
    export LD_LIBRARY_PATH=../libs/linux32
fi

echo "--> ClassPath: $CP"

SRC_LIB=main/java/com/netsdk/lib/*.java 
SRC_COMMON=main/java/com/netsdk/common/*.java 
SRC_MODULE=main/java/com/netsdk/demo/module/*.java 
SRC_FRAME=main/java/com/netsdk/demo/frame/*.java
SRC_ATTENDANCE=main/java/com/netsdk/demo/frame/Attendance/*.java
SRC_AUTOREGISTER=main/java/com/netsdk/demo/frame/AutoRegister/*.java
SRC_FACERECOGNITION=main/java/com/netsdk/demo/frame/FaceRecognition/*.java
SRC_GATE=main/java/com/netsdk/demo/frame/Gate/*.java
SRC_THERMALCAMERA=main/java/com/netsdk/demo/frame/ThermalCamera/*.java

if [ ! -d "../bin" ]; then
 mkdir ../bin
fi

BIN+=../bin
echo "--> Bin $BIN"

javac -cp $CP $SRC_LIB $SRC_COMMON $SRC_MODULE $SRC_FRAME $SRC_ATTENDANCE $SRC_AUTOREGISTER $SRC_FACERECOGNITION $SRC_GATE $SRC_THERMALCAMERA -d $BIN 


#运行
cd ../bin
echo "--> path:" pwd

if [ "$1" == "-frame" ]; then
   DEMO=main/java/com/netsdk/demo/frame/$2
   echo "--> example name: $DEMO"
elif [ "$1" == "-module" ]; then
   DEMO=main/java/com/netsdk/demo/module/$2
   echo "--> snippet name: $DEMO"
else
   temp=$1
   temp=${temp:1}
   DEMO=main/java/com/netsdk/demo/frame/$temp/$2
   echo "--> name: $DEMO"
fi

java -cp $CP:. $DEMO

cd -
