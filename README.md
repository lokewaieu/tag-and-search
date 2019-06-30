# Cross Camera Person Tag & Search
This is an early attempt of a real life usage of a person search system. Person Search is loosely defined as a system where a camera is able to "remember" a newly seen person into its "memory" (known as Gallery) and subsequently, able to reidentify the same person if he is seen again. The method used in this system is a 2 step approach: First performing person detection, then performing person re-identification.

Main features of this system include:
* Real-time deep learning detection and re-identification inferencing using ONLY Intel Processors (~40fps+)
* Web-based user interface for initiating person search, powered by a person detection backend for the selection of the query image
* Easily scalable to accomodate more cameras (2 cameras used in this)

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
The location of the source file relevant to this project can be found here, replace the main.cpp with that provided in this repository:
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

```


