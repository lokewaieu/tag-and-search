# Cross Camera Person Tag & Search using Person Detection and Person Re-identification
This is an early attempt of a real life usage of a person search system. Person Search is loosely defined as a system where a camera is able to "remember" a newly seen person into its "memory" (known as Gallery) and subsequently, able to reidentify the same person if he is seen again. The method used in this system is a 2 step approach: First performing person detection, then performing person re-identification.

The system consists of 3 basic components:
* 1 Server
* 1 or more cameras
* 1 UI

Main features of this system include:
* Fast: Real-time deep learning detection and re-identification inferencing using ONLY Intel Processors (~40fps+)
* User Friendly: AI-Enable browser-based user interface (AUI) for initiating person search, powered by a person detection backend for the selection of the query image
* Scalable: Easily add more cameras (2 cameras used in this)
* Flexible: Cameras and server can all be setup on a single PC, or individual PC for each component

# Prerequisites
1. Intel OpenVINO Toolkit: https://software.intel.com/en-us/openvino-toolkit/choose-download
2. Node-Red MQTT Broker: https://nodered.org/docs/getting-started/local
3. Eclipse Paho MQTT for C++: https://github.com/eclipse/paho.mqtt.cpp
4. Base64 encoder for C++: https://github.com/ReneNyffenegger/cpp-base64
5. At least 2 IP/USB cameras

# Setting up Node-Red
After installing node-red, add/install the following MOSCA MQTT pallete within the node-red user interface, accessed via localhost:1880 :-
```
node-red-contrib-mqtt-broker
```
Node-Red can be started by entering the following into the terminal (needed for communication between cameras,server and UI):
```
$ node-red
```

# Usage on Linux (Ubuntu 16.04++)
Properly install Intel OpenVINO toolkit according to the guide provided by the official website. Installation on ROOT path is highly recommended. Proceed the installation by setting and building all relevant dependencies by continuing the tutorial provided directly by Intel OpenVINO (Important).

Since the toolkit only provides the bare minimum in terms of the re-identification source code, this is where you need to copy and paste the source code provided in this repository to the following path, and build it.

The location of the source file relevant to this project can be found here, replace the main.cpp with that provided in this repository AND copy the header file for base64 encoding (base64.h) into the following folder:
```
$ cd /opt/intel/computer_vision_sdk/inference_engine/samples/crossroad_camera_demo/
```
The executable application can be built by doing so (build for all 3 source codes: cam1, cam2 and server):
```
$ cd /opt/intel/computer_vision_sdk/inference_engine/samples/crossroad_camera_demo/build/
$ make
```
Run each executable individually.
Camera 1:
```
./cam_1 -i <IP-camera-address||Default 0 if USB cam> -m /opt/intel/computer_vision_sdk/deployment_tools/intel_m
odels/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -m_reid /opt/intel/computer_vision_sdk/deployment_tools/intel_models/person-reidentification-retail-0079/FP32/person-reidentification-retail-0079.xml -t 0.8 -t_reid 0.8
```
Camera 2:
```
./cam_2 -i <IP-camera-address||Default 0 if USB cam> -m /opt/intel/computer_vision_sdk/deployment_tools/intel_m
odels/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -m_reid /opt/intel/computer_vision_sdk/deployment_tools/intel_models/person-reidentification-retail-0079/FP32/person-reidentification-retail-0079.xml -t 0.8 -t_reid 0.8
```
Server:
```
./server -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -m_reid /opt/intel/computer_vision_sdk/deployment_tools/intel_models/person-reidentification-retail-0079/FP32/person-reidentification-retail-0079.xml
```
# Using the AUI
The AUI can be found here:
```
$ /surv_UI/index.html
```
IP address for MQTT can be changed here:
```
$ /surv_UI/mqtt2.js
```
Steps for navigating through the AUI:
* Click pause when the person-of-interest comes into view
* Hover mouse above the person-of-interest, a green bounding box should appear, highlighting the person
* Click once on the person, this is to send the query image to the server in preparation for person re-identification
* Then click Confirm to initiate person re-identification
* Click Results to display search results, the results only show the top 3 matches from each camera
* Person Search process is successfully completed
