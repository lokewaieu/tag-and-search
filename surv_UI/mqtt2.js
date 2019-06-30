var mqtt;
var reconnectTimeout = 2000;
var host = "192.168.1.102";//change this
var port = 1884;
var livestream;
var frame;
var stream_camera = "1";
var result_number = 0;
var one_time = true;

function onConnect(){
	console.log("Connected");
    mqtt.subscribe("test_video");
    mqtt.subscribe("Camera_1");
	mqtt.subscribe("Camera_2");
	mqtt.subscribe("cropped_image");
	mqtt.subscribe("top_first_cam_1");
    mqtt.subscribe("top_second_cam_1");
    mqtt.subscribe("top_third_cam_1");
    mqtt.subscribe("top_first_cam_2");
    mqtt.subscribe("top_second_cam_2");
    mqtt.subscribe("top_third_cam_2");
	mqtt.subscribe("stream_1");
	mqtt.subscribe("stream_2");
	mqtt.subscribe("display_ui");

	$(".notification-bell").removeClass('animated');
	$(".notification-count").removeClass('animated');
}

function MQTTconnect(){
	console.log("Connecting to " + host + ":" + port);
	mqtt = new Paho.MQTT.Client(host,port,"");
	var options = {
		timeout: 3,
		onSuccess: onConnect,
		onFailure: onFailure,
	};
	mqtt.onMessageArrived = onMessageArrived;
	mqtt.connect(options);
}

function onFailure(message){
	console.log("Connection Attempt to Host " + host + "Failed");
	setTimeout(MQTTconnect,reconnectTimeout)

}

function onMessageArrived(msg){

	if (msg.destinationName == "cropped_image") {
		var img = document.getElementById("Lobby_1");
		var img_2 = document.getElementById("Carpark_1");
		img.src = 'data:image/jpeg;base64,' + msg.payloadString;
		img_2.src = 'data:image/jpeg;base64,' + msg.payloadString;
	}
	if (msg.destinationName == "top_first_cam_1"){
		var img = document.getElementById("Lobby_2");
		showAlertText();
		result_number++;
		img.src = 'data:image/jpeg;base64,' + msg.payloadString;
	}

	if (msg.destinationName == "top_second_cam_1"){
		var img = document.getElementById("Lobby_3");
		result_number++;
		img.src = 'data:image/jpeg;base64,' + msg.payloadString;
	}

	if (msg.destinationName == "top_third_cam_1"){
		var img = document.getElementById("Lobby_4");
		result_number++;
		img.src = 'data:image/jpeg;base64,' + msg.payloadString;
	}

	if (msg.destinationName == "top_first_cam_2"){
		var img = document.getElementById("Carpark_2");
		showAlertText();
		result_number++;
		img.src = 'data:image/jpeg;base64,' + msg.payloadString;
	}

	if (msg.destinationName == "top_second_cam_2"){
		var img = document.getElementById("Carpark_3");
		result_number++;
		img.src = 'data:image/jpeg;base64,' + msg.payloadString;
	}

	if (msg.destinationName == "top_third_cam_2"){
		var img = document.getElementById("Carpark_4");
		result_number++;
		img.src = 'data:image/jpeg;base64,' + msg.payloadString;
	}
	if (msg.destinationName == "display_ui") {
		console.log("display_ui");
		var img = document.getElementById("img1");
		img.src = 'data:image/jpeg;base64,' + msg.payloadString;
	}
	if (stream_camera == "1") {
		if (msg.destinationName == "stream_1") {
			var video = document.getElementById("video");
			document.getElementsByClassName("cam_title")[0].innerHTML = "Camera 1";
			document.getElementsByClassName("area_title")[0].innerHTML = "LAB BACK";
			video.src = 'data:image/jpeg;base64,' + msg.payloadString;
		}
	}	
	if (stream_camera == "2") {
		if (msg.destinationName == "stream_2") {
			var video = document.getElementById("video");
			document.getElementsByClassName("cam_title")[0].innerHTML = "Camera 2";
			document.getElementsByClassName("area_title")[0].innerHTML = "LAB FRONT";
			video.src = 'data:image/jpeg;base64,' + msg.payloadString;
		}
	}

	if (one_time == true) {
		if (msg.destinationName == "stream_1" || msg.destinationName == "stream_2") {
			if (msg.destinationName == "stream_1") {
				var camWin1 = document.getElementById("cam_window_1");
				camWin1.src = 'data:image/jpeg;base64,' + msg.payloadString;
			}
			if (msg.destinationName == "stream_2") {
				var camWin2 = document.getElementById("cam_window_2");
				camWin2.src = 'data:image/jpeg;base64,' + msg.payloadString;
			}
		}
		one_time = false;
	}
}

function stream_1() {
	stream_camera = "1";
	document.getElementsByClassName("cam_title")[0].innerHTML = "Camera 1";
	document.getElementsByClassName("area_title")[0].innerHTML = "LAB BACK";
	one_time = true;
	//console.log("Receving stream from camera: " + stream_camera);
}

function stream_2() {
	stream_camera = "2";
	document.getElementsByClassName("cam_title")[0].innerHTML = "Camera 2";
	document.getElementsByClassName("area_title")[0].innerHTML = "LAB FRONT";
	one_time = true;
	//console.log("Receiving stream from camera: " + stream_camera);
}

function cycle_cam() {
	console.log("cycle");
	if (stream_camera == "1") {
		stream_2();
	}
	else if (stream_camera == "2") {
		stream_1();
	}
}

function confirm(){
	message = new Paho.MQTT.Message("");
    message.destinationName = "ok_signal";
	mqtt.send(message);
	modal.style.display = "none";

	setTimeout(function (){
		document.getElementsByClassName("notification-count")[0].innerHTML = result_number;
	}, 1000); // How long do you want the delay to be (in milliseconds)? 
	
}

function showAlertText() {
	$(".notification-bell").addClass('animated');
	$(".notification-count").addClass('animated');
	$(".alertTextAnim").addClass('animated');
}

window.onload = function(){
	var video = document.getElementById("video");
	setInterval(updateClock, 1000);
	MQTTconnect();
}

function pauseVid() {

	var vid = document.getElementById("video");
	ctx = canvas.getContext('2d');
    ctx.drawImage(vid, 0,0, 728.9,410); 
  	var img = document.getElementById("img1");
  	var result = canvas.toDataURL();
  	// console.log(result);
	var slice = result.split(',')[1];
	//console.log(slice);
	img.src = 'data:image/jpeg;base64,' + slice;
  	message = new Paho.MQTT.Message(slice);
    message.destinationName = "query_image";
	mqtt.send(message);
  	var modal = document.getElementById("myModal");
  	modal.style.display = "block";
}