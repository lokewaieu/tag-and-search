/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

/**
* \brief The entry point for the Inference Engine crossroad_camera sample application
* \file crossroad_camera_sample/main.cpp
* \example crossroad_camera_sample/main.cpp
*/

/* v1.0 18-1-2019
   v1.1 22-1-2019 - Double Cam REID
   v1.5 23-1-2019 - Search Mode added
   v1.6 24-1-2019 - Added Smart UI function
   v1.7 27-1-2019 - Added Ring Buffer for video recording'
   v1.8 10-4-2019 - Fixed memory issues  
   v1.9 18-4-2019 - Changed tagging algorithm. Tagging MIGHT not take off at times*/

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <inference_engine.hpp>
#include <samples/common.hpp>
#include "crossroad_camera_sample.hpp"
#include <ext_list.hpp>

#include <opencv2/opencv.hpp>
#include "mqtt/async_client.h"
#include "base64.h"

#define ROOT "/home/ubuntu/Desktop/"

using namespace cv;
using namespace std;
using namespace InferenceEngine;

static const string base64_chars = 
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

const string DFLT_SERVER_ADDRESS    {"192.168.1.102:1883"};
const string DFLT_CLIENT_ID         {"SmartCam_2_send"};
const string TOPIC_IMAGE            {"Camera_2"};
const string TOPIC_FEATURE          {"Feature_Camera_2"};
const string TOPIC_SEARCH_START     {"search_mode_start"};
const string TOPIC_SEARCH_STOP      {"search_mode_stop"};
const string TOPIC_QUERY_VECTOR     {"query_vector"};
const string TOPIC_STREAM           {"stream_2"};
const string TOPIC_CROPPED_IMAGE    {"cropped_image"};
const string TOPIC_OK_SIGNAL        {"ok_signal"};
const string TOPIC_TOP_1_CAM2       {"top_first_cam_2"};
const string TOPIC_TOP_2_CAM2       {"top_second_cam_2"};
const string TOPIC_TOP_3_CAM2       {"top_third_cam_2"};
const string TOPIC_CLOSE_WINDOWS    {"close_window"};

const char* LWT_PAYLOAD = "\nI'm dying unexpectedly. Bye :(\n";

const int QOS = 1;
const auto TIMEOUT = chrono::seconds(5);
const int  N_RETRY_ATTEMPTS = 999;

bool close_windows = false;
bool getCroppedImage = false;
bool isSearchMode = false;
bool OK_signal = false;
bool quit = false;
bool reid_enable = false;
bool targetQueryLocated = false;
bool top_1 = false;
bool top_2 = false;
bool top_3 = false;

Mat query_image;
Mat cropped_image;

vector<vector<float>> searchReIdVec;
vector<vector<uchar>> imageDB;

int j = 0;
int sleep_time = 10;
float top_3_score[3];

string  address     = DFLT_SERVER_ADDRESS,
        clientID    = DFLT_CLIENT_ID;

mqtt::async_client client(address, clientID);

// -------------------------A base action listener------------------------------------

class action_listener : public virtual mqtt::iaction_listener
{
    string name_;

    void on_failure(const mqtt::token& tok) override {
        cout << name_ << " failure";
        if (tok.get_message_id() != 0)
            cout << " for token: [" << tok.get_message_id() << "]" << endl;
        cout << endl;
    }

    void on_success(const mqtt::token& tok) override {
        cout << name_ << " success";
        if (tok.get_message_id() != 0)
            cout << " for token: [" << tok.get_message_id() << "]" << endl;
        auto top = tok.get_topics();
        if (top && !top->empty())
            cout << "\ttoken topic: '" << (*top)[0] << "', ..." << endl;
        cout << endl;
    }

public:
    action_listener(const string& name) : name_(name) {}
};

/////////////////////////////////////////////////////////////////////////////

/**
 * Local callback & listener class for use with the client connection.
 * This is primarily intended to receive messages, but it will also monitor
 * the connection to the broker. If the connection is lost, it will attempt
 * to restore the connection and re-subscribe to the topic.
 */
class callback : public virtual mqtt::callback,
                    public virtual mqtt::iaction_listener
{
    // Counter for the number of connection retries
    int nretry_;
    // The MQTT client
    mqtt::async_client& cli_;
    // Options to use if we need to reconnect
    mqtt::connect_options& connOpts_;
    // An action listener to display the result of actions.
    action_listener subListener_;

    // This deomonstrates manually reconnecting to the broker by calling
    // connect() again. This is a possibility for an application that keeps
    // a copy of it's original connect_options, or if the app wants to
    // reconnect with different options.
    // Another way this can be done manually, if using the same options, is
    // to just call the async_client::reconnect() method.
    void reconnect() {
        this_thread::sleep_for(chrono::milliseconds(2500));
        try {
            cli_.connect(connOpts_, nullptr, *this);
        }
        catch (const mqtt::exception& exc) {
            cerr << "Error: " << exc.what() << endl;
            exit(1);
        }
    }

    // Re-connection failure
    void on_failure(const mqtt::token& tok) override {
        cout << "Connection attempt failed" << endl;
        if (++nretry_ > N_RETRY_ATTEMPTS)
            exit(1);
        reconnect();
    }

    // (Re)connection success
    // Either this or connected() can be used for callbacks.
    void on_success(const mqtt::token& tok) override {}

    // (Re)connection success
    void connected(const string& cause) override {
        cout << "\nConnection success" << endl;
        // cout << "\nSubscribing to topic '" << TOPIC << "'\n"
        //  << "\tfor client " << CLIENT_ID
        //  << " using QoS" << QOS << "\n"
        //  << "\nPress Q<Enter> to quit\n" << endl;

        cli_.subscribe(TOPIC_SEARCH_START, QOS, nullptr, subListener_);
        cli_.subscribe(TOPIC_SEARCH_STOP, QOS, nullptr, subListener_);
        cli_.subscribe(TOPIC_QUERY_VECTOR, QOS, nullptr, subListener_);
        cli_.subscribe(TOPIC_OK_SIGNAL, QOS, nullptr, subListener_);
        cli_.subscribe(TOPIC_CROPPED_IMAGE, QOS, nullptr, subListener_);
        cli_.subscribe(TOPIC_CLOSE_WINDOWS, QOS, nullptr, subListener_);
    }

    // Callback for when the connection is lost.
    // This will initiate the attempt to manually reconnect.
    void connection_lost(const string& cause) override {
        cout << "\nConnection lost" << endl;
        if (!cause.empty())
            cout << "\tcause: " << cause << endl;

        cout << "Reconnecting..." << endl;
        nretry_ = 0;
        reconnect();
    }

    // Callback for when a message arrives.
    void message_arrived(mqtt::const_message_ptr msg) override {

        if (msg->get_topic() == "search_mode_start") {
            cout << "[ MQTT ] Request for Search Mode received\n";

            searchReIdVec.clear();
            isSearchMode = true;
        }
        if (msg->get_topic() == "search_mode_stop") {
            cout << "[ MQTT ] Request for Tagging Mode received\n";

            isSearchMode = false;
        }
        if (msg->get_topic() == "query_vector") {
            cout << "[ MQTT ] Receiving query vector from server\n";

            string decodedVec_asString = msg->to_string();
            vector<float> decodedVec_asFloat;

            stringstream stream(decodedVec_asString);

            while(stream.good()) {
                string substr;
                float substr_f;

                getline(stream, substr, ',');
                istringstream in_stream(substr);
                copy(istream_iterator<float>(in_stream), istream_iterator<float>(), back_inserter(decodedVec_asFloat));
            }
            searchReIdVec.push_back(decodedVec_asFloat);
        }
        if (msg->get_topic() == "cropped_image") {
            cout << "[ MQTT ] Receiving cropped image from server\n";

            string image_asString = msg->to_string();

            image_asString = base64_decode(image_asString);

            vector<uchar> data(image_asString.begin(), image_asString.end());
            cropped_image = imdecode(data, 1);

            getCroppedImage = true;
        }
        if (msg->get_topic() == "ok_signal") {
            cout << "[ MQTT ] Re-id START request received\n";

            OK_signal = true;
        }
        if (msg->get_topic() == "close_window") {
            cout << "[ MQTT ] Close window requested by UI\n";

            close_windows = true;
        }
    }

    void delivery_complete(mqtt::delivery_token_ptr token) override {}

public:
    callback(mqtt::async_client& cli, mqtt::connect_options& connOpts)
                : nretry_(0), cli_(cli), connOpts_(connOpts), subListener_("Subscription") {}
};

// ------------------------A derived action listener for publish events---------------------

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    cout << "[ INFO ] Parsing input parameters" << endl;

    if (FLAGS_i.empty()) {
        throw logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw logic_error("Parameter -m is not set");
    }

    return true;
}

// -------------------------Generic routines for detection networks-------------------------------------------------
struct BaseDetection {
    ExecutableNetwork net;
    InferencePlugin plugin;
    InferRequest request;
    string & commandLineFlag;
    string topoName;
    Blob::Ptr inputBlob;
    string inputName;
    string outputName;

    BaseDetection(string &commandLineFlag, string topoName)
            : commandLineFlag(commandLineFlag), topoName(topoName) {}

    ExecutableNetwork * operator ->() {
        return &net;
    }
    virtual CNNNetwork read()  = 0;

    virtual void setRoiBlob(const Blob::Ptr &roiBlob) {
        if (!enabled())
            return;
        if (!request)
            request = net.CreateInferRequest();

        request.SetBlob(inputName, roiBlob);
    }

    virtual void enqueue(const Mat &person) {
        if (!enabled())
            return;
        if (!request)
            request = net.CreateInferRequest();

        if (FLAGS_auto_resize) {
            inputBlob = wrapMat2Blob(person);
            request.SetBlob(inputName, inputBlob);
        } else {
            inputBlob = request.GetBlob(inputName);
            matU8ToBlob<uint8_t>(person, inputBlob);
        }
    }

    virtual void submitRequest() {
        if (!enabled() || !request) return;
        request.StartAsync();
    }

    virtual void wait() {
        if (!enabled()|| !request) return;
        request.Wait(IInferRequest::WaitMode::RESULT_READY);
    }
    mutable bool enablingChecked = false;
    mutable bool _enabled = false;

    bool enabled() const  {
        if (!enablingChecked) {
            _enabled = !commandLineFlag.empty();
            if (!_enabled) {
                cout << "[ INFO ] " << topoName << " detection DISABLED" << endl;
            }
            enablingChecked = true;
        }
        return _enabled;
    }
};

void find_top_3(float arr[], int arr_size, double FLAGS_t_reid) {

    int i;
    float first, second, third;

    if (arr_size < 3) {
        cout << "Invalid input. Size of " << arr_size << endl;
        return;
    }

    third = first = second = INT_MIN;

    cout << "From cosSim" << endl;

    for (i = 0; i < arr_size; ++i) {
        cout << fixed << arr[i] << endl;
    }

    for (i = 0; i < arr_size; ++i) {

        if (arr[i] > first) {

            third = second;
            second = first;
            first = arr[i];

            top_3_score[0] = i;
        }
        else if (arr[i] > second) {

            third = second;
            second = arr[i];

            top_3_score[1] = i;
        }
        else if (arr[i] > third) {

            third = arr[i];

            top_3_score[2] = i;
        }
    }

    for (i = 0; i < arr_size; ++i) {

        if (arr[i] == first) {

            top_3_score[0] = i;

            if (arr[i] > 0.4) {

                top_1 = true;
            }
        }
        if (arr[i] == second) {

            top_3_score[1] = i;

            if (arr[i] > 0.4) {

                top_2 = true;
            }
        }
        if (arr[i] == third) {

            top_3_score[2] = i;

            if (arr[i] > 0.4) {

                top_3 = true;
            }
        }
    }

    cout << "[ INFO ] Top 3 scores: " << first << ", " << second << ", " << third << endl;
    cout << "[ INFO ] Top 3 indexes: " << top_3_score[0] << ", " << top_3_score[1] << ", " << top_3_score[2] << endl;
}

struct PersonDetection : BaseDetection{
    int maxProposalCount;
    int objectSize;
    float width = 0;
    float height = 0;
    bool resultsFetched = false;

    struct Result {
        int label;
        float confidence;
        Rect location;
    };

    vector<Result> results;

    void submitRequest() override {
        resultsFetched = false;
        results.clear();
        BaseDetection::submitRequest();
    }

    void setRoiBlob(const Blob::Ptr &frameBlob) override {
        height = frameBlob->getTensorDesc().getDims()[2];
        width = frameBlob->getTensorDesc().getDims()[3];
        BaseDetection::setRoiBlob(frameBlob);
    }

    void enqueue(const Mat &frame) override {
        height = frame.rows;
        width = frame.cols;
        BaseDetection::enqueue(frame);
    }

    PersonDetection() : BaseDetection(FLAGS_m, "Person Detection") {}
    CNNNetwork read() override {
        cout << "\n[ INFO ] Loading network files for PersonDetection" << endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Set batch size to 1 **/
        cout << "[ INFO ] Batch size is forced to  1" << endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extract model name and load it's weights **/
        string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        // -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        cout << "[ INFO ] Checking Person Detection inputs" << endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw logic_error("Person Detection network should have only one input");
        }
        InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setInputPrecision(Precision::U8);

        if (FLAGS_auto_resize) {
            inputInfoFirst->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            inputInfoFirst->getInputData()->setLayout(Layout::NHWC);
        } else {
            inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        }
        inputName = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        cout << "[ INFO ] Checking Person Detection outputs" << endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw logic_error("Person Detection network should have only one output");
        }
        DataPtr& _output = outputInfo.begin()->second;
        const SizeVector outputDims = _output->getTensorDesc().getDims();
        outputName = outputInfo.begin()->first;
        maxProposalCount = outputDims[2];
        objectSize = outputDims[3];
        if (objectSize != 7) {
            throw logic_error("Output should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw logic_error("Incorrect output dimensions for SSD");
        }
        _output->setPrecision(Precision::FP32);
        _output->setLayout(Layout::NCHW);

        cout << "[ INFO ] Loading Person Detection model to the "<< FLAGS_d << " plugin" << endl;
        return netReader.getNetwork();
    }

    void fetchResults() {
        if (!enabled()) return;
        results.clear();
        if (resultsFetched) return;
        resultsFetched = true;
        const float *detections = request.GetBlob(outputName)->buffer().as<float *>();
        // pretty much regular SSD post-processing
        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];  // in case of batch
            if (image_id < 0) {  // indicates end of detections
                break;
            }

            Result r;
            r.label = static_cast<int>(detections[i * objectSize + 1]);
            r.confidence = detections[i * objectSize + 2];

            r.location.x = detections[i * objectSize + 3] * width;
            r.location.y = detections[i * objectSize + 4] * height;
            r.location.width = detections[i * objectSize + 5] * width - r.location.x;
            r.location.height = detections[i * objectSize + 6] * height - r.location.y;

            if (FLAGS_r) {
                cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                          "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
                          << r.location.height << ")"
                          << ((r.confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << endl;
            }

            if (r.confidence <= FLAGS_t) {
                continue;
            }
            results.push_back(r);
        }
    }
};

struct PersonReIdentification : BaseDetection {
    vector<vector<float>> globalReIdVec;  // contains vectors characterising all detected persons

    PersonReIdentification() : BaseDetection(FLAGS_m_reid, "Person Reidentification Retail") {}

    unsigned long int findMatchingPerson(const vector<float> &newReIdVec) {
        float cosSim;
        auto size = globalReIdVec.size();
        float cosSimArr[size];
        vector<uchar> buf;

        if (reid_enable) {

            for (auto i = 0; i < size; ++i) {

                cosSim = cosineSimilarity(newReIdVec, globalReIdVec[i]);
                cout << "[ REID ] cosineSimilarity: " << cosSim << endl;

                if (true) { //cosSim > FLAGS_t_reid

                    cosSimArr[i] = cosSim;
                }
            }

            cout << "[ INFO ] Size of " << size << endl;

            if (size >= 3) {
                cout << "[ INFO ] Collecting top 3 matches" << endl;
                find_top_3(cosSimArr, size, FLAGS_t_reid);
                int size_n = sizeof(top_3_score)/sizeof(top_3_score[0]);

                cout << "Size of imageDB: " << imageDB.size() << endl;

                for (int i = 0; i < size_n; i++) {

                    if (!imageDB[top_3_score[i]].empty() && i == 0 && top_1) {

                        Mat img = imdecode(imageDB[top_3_score[i]], 1);

                        imencode(".jpg", img, buf);
                        uchar *enc_msg = new uchar[buf.size()];

                        for (int i = 0; i < buf.size(); i++) {
                            enc_msg[i] = buf[i];
                        }
                        string encoded = base64_encode(enc_msg, buf.size());

                        try {
                            cout << "[ MQTT ] Sending potential match to UI";
                            mqtt::delivery_token_ptr pubtok;
                            pubtok = client.publish(TOPIC_TOP_1_CAM2, encoded, QOS, false);
                            cout << " with token ID " << pubtok->get_message_id() << " for size of "
                                << pubtok->get_message()->get_payload().size() << " bytes" << endl;
                        }
                        catch (const mqtt::exception& exc) {
                            cerr << exc.what() << endl;
                            continue;
                        }

                        top_1 = false;
                    }
                    else if (!imageDB[top_3_score[i]].empty() && i == 1 && top_2) {

                        Mat img = imdecode(imageDB[top_3_score[i]], 1);

                        imencode(".jpg", img, buf);
                        uchar *enc_msg = new uchar[buf.size()];

                        for (int i = 0; i < buf.size(); i++) {
                            enc_msg[i] = buf[i];
                        }
                        string encoded = base64_encode(enc_msg, buf.size());

                        try {
                            cout << "[ MQTT ] Sending potential match to UI";
                            mqtt::delivery_token_ptr pubtok;
                            pubtok = client.publish(TOPIC_TOP_2_CAM2, encoded, QOS, false);
                            cout << " with token ID " << pubtok->get_message_id() << " for size of "
                                << pubtok->get_message()->get_payload().size() << " bytes" << endl;
                        }
                        catch (const mqtt::exception& exc) {
                            cerr << exc.what() << endl;
                            continue;
                        }

                        top_2 = false;
                    }
                    else if (!imageDB[top_3_score[i]].empty() && i == 2 && top_3) {

                        Mat img = imdecode(imageDB[top_3_score[i]], 1);

                        imencode(".jpg", img, buf);
                        uchar *enc_msg = new uchar[buf.size()];

                        for (int i = 0; i < buf.size(); i++) {
                            enc_msg[i] = buf[i];
                        }
                        string encoded = base64_encode(enc_msg, buf.size());

                        try {
                            cout << "[ MQTT ] Sending potential match to UI";
                            mqtt::delivery_token_ptr pubtok;
                            pubtok = client.publish(TOPIC_TOP_3_CAM2, encoded, QOS, false);
                            cout << " with token ID " << pubtok->get_message_id() << " for size of "
                                << pubtok->get_message()->get_payload().size() << " bytes" << endl;
                        }
                        catch (const mqtt::exception& exc) {
                            cerr << exc.what() << endl;
                            continue;
                        }

                        top_3 = false;
                    }
                }
            }
        }
        else if (!isSearchMode) {
            /* assigned REID is index of the matched vector from the globalReIdVec */
            for (auto i = 0; i < size; ++i) {

                cosSim = cosineSimilarity(newReIdVec, globalReIdVec[i]);
                //cout << "cosineSimilarity: " << cosSim << endl;

                if (FLAGS_r) {
                    cout << "cosineSimilarity: " << cosSim << endl;
                }
                if (cosSim > FLAGS_t_reid) {
                    /* We substitute previous person's vector by a new one characterising
                     * last person's position */

                    globalReIdVec[i] = newReIdVec;

                    return i;
                }
            }

            globalReIdVec.push_back(newReIdVec);

            try {

                stringstream ss1;

                for (size_t i = 0; i < newReIdVec.size(); i++) {
                    ss1 << "\n";
                    ss1 << newReIdVec[i];
                }

                string s = ss1.str();

                //cout << "[ MQTT ] Sending vector feature to server ";
                //mqtt::delivery_token_ptr pubtok;
                //pubtok = client.publish(TOPIC_FEATURE, s, QOS, false);
                //cout << "with token ID " << pubtok->get_message_id() << " for size of "
                //    << pubtok->get_message()->get_payload().size() << " bytes\n";
                    //pubtok->wait_for(TIMEOUT);
            }
            catch (const mqtt::exception& exc) {
                cerr << exc.what() << endl;
            }
        }
        else if (isSearchMode) {

            for (auto i = 0; i < searchReIdVec.size(); ++i) {

                cosSim = cosineSimilarity(newReIdVec, searchReIdVec[i]);
                cout << "[ SEARCH ] cosineSimilarity: " << cosSim << endl;

                // ofstream outfile;
                // outfile.open("/home/ubuntu/Desktop/log.txt", ios_base::app);
                // outfile << cosSim << endl;

                if (FLAGS_r) {
                    cout << "cosineSimilarity: " << cosSim << endl;
                }
                if (cosSim > FLAGS_t_reid) {
                    /* We substitute previous person's vector by a new one characterising
                     * last person's position */

                    searchReIdVec[i] = newReIdVec;
                    targetQueryLocated = true;

                    return i;
                }
                else {
                    targetQueryLocated = false;
                }
            }
        }
        return size;
    }

    vector<float> getReidVec() {
        Blob::Ptr attribsBlob = request.GetBlob(outputName);

        auto numOfChannels = attribsBlob->getTensorDesc().getDims().at(1);
        /* output descriptor of Person Reidentification Recognition network has size 256 */
        if (numOfChannels != 256) {
            throw logic_error("Output size (" + to_string(numOfChannels) + ") of the "
                                   "Person Reidentification network is not equal to 256");
        }

        auto outputValues = attribsBlob->buffer().as<float*>();
            return vector<float>(outputValues, outputValues + 256);
    }

    template <typename T>
    float cosineSimilarity(const vector<T> &vecA, const vector<T> &vecB) {
        if (vecA.size() != vecB.size()) {
            throw logic_error("cosine similarity can't be called for the vectors of different lengths: "
                                   "vecA size = " + to_string(vecA.size()) +
                                   "vecB size = " + to_string(vecB.size()));
        }

        T mul, denomA, denomB, A, B;
        mul = denomA = denomB = A = B = 0;
        for (auto i = 0; i < vecA.size(); ++i) {
            A = vecA[i];
            B = vecB[i];
            mul += A * B;
            denomA += A * A;
            denomB += B * B;
        }
        if (denomA == 0 || denomB == 0) {
            throw logic_error("cosine similarity is not defined whenever one or both "
                                   "input vectors are zero-vectors.");
        }
        return mul / (sqrt(denomA) * sqrt(denomB));
    }

    CNNNetwork read() override {
        cout << "[ INFO ] Loading network files for Person Reidentification" << endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_reid);
        cout << "[ INFO ] Batch size is forced to  1 for Person Reidentification Network" << endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extract model name and load it's weights **/
        string binFileName = fileNameNoExt(FLAGS_m_reid) + ".bin";
        netReader.ReadWeights(binFileName);

        /** Person Reidentification network should have 1 input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        cout << "[ INFO ] Checking Person Reidentification Network input" << endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw logic_error("Person Reidentification Retail should have 1 input");
        }
        InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setInputPrecision(Precision::U8);
        if (FLAGS_auto_resize) {
            inputInfoFirst->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            inputInfoFirst->getInputData()->setLayout(Layout::NHWC);
        } else {
            inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        }
        inputName = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        cout << "[ INFO ] Checking Person Reidentification Network output" << endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw logic_error("Person Reidentification Network should have 1 output");
        }
        outputName = outputInfo.begin()->first;
        cout << "[ INFO ] Loading Person Reidentification Retail model to the "<< FLAGS_d_reid << " plugin" << endl;

        _enabled = true;
        return netReader.getNetwork();
    }
};

struct Load {
    BaseDetection& detector;
    explicit Load(BaseDetection& detector) : detector(detector) { }

    void into(InferencePlugin & plg) const {
        if (detector.enabled()) {
            detector.net = plg.LoadNetwork(detector.read(), {});
            detector.plugin = plg;
        }
    }
};

static inline bool is_base64(unsigned char c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
  string ret;
  int i = 0;
  int j = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (in_len--) {
    char_array_3[i++] = *(bytes_to_encode++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for(i = 0; (i <4) ; i++)
        ret += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

  if (i)
  {
    for(j = i; j < 3; j++)
      char_array_3[j] = '\0';

    char_array_4[0] = ( char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

    for (j = 0; (j < i + 1); j++)
      ret += base64_chars[char_array_4[j]];

    while((i++ < 3))
      ret += '=';

  }

  return ret;
}

string base64_decode(string const& encoded_string) {
  int in_len = encoded_string.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  unsigned char char_array_4[4], char_array_3[3];
  string ret;

  while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
    char_array_4[i++] = encoded_string[in_]; in_++;
    if (i ==4) {
      for (i = 0; i <4; i++)
        char_array_4[i] = base64_chars.find(char_array_4[i]);

      char_array_3[0] = ( char_array_4[0] << 2       ) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

      for (i = 0; (i < 3); i++)
        ret += char_array_3[i];
      i = 0;
    }
  }

  if (i) {
    for (j = 0; j < i; j++)
      char_array_4[j] = base64_chars.find(char_array_4[j]);

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

    for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
  }

  return ret;
}

template <class T>
class circular_buffer {
public:
    explicit circular_buffer(size_t size) :
        buf_(std::unique_ptr<T[]>(new T[size])),
        max_size_(size)
    {

    }

    void put(T item)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        buf_[head_] = item;

        if(full_)
        {
            tail_ = (tail_ + 1) % max_size_;
        }

        head_ = (head_ + 1) % max_size_;

        full_ = head_ == tail_;
    }

    T get()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if(empty())
        {
            return T();
        }

        //Read data and advance the tail (we now have a free space)
        auto val = buf_[tail_];
        full_ = false;
        tail_ = (tail_ + 1) % max_size_;

        return val;
    }

    void reset()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        head_ = tail_;
        full_ = false;
    }

    bool empty() const
    {
        //if head and tail are equal, we are empty
        return (!full_ && (head_ == tail_));
    }

    bool full() const
    {
        //If tail is ahead the head by 1, we are full
        return full_;
    }

    size_t capacity() const
    {
        return max_size_;
    }

    size_t size() const
    {
        size_t size = max_size_;

        if(!full_)
        {
            if(head_ >= tail_)
            {
                size = head_ - tail_;
            }
            else
            {
                size = max_size_ + head_ - tail_;
            }
        }

        return size;
    }

private:
    std::mutex mutex_;
    std::unique_ptr<T[]> buf_;
    size_t head_ = 0;
    size_t tail_ = 0;
    const size_t max_size_;
    bool full_ = 0;
};

circular_buffer<Mat> ring_buffer(32);

void framePusher_() {

    int i = 0;
    vector<uchar> buf;
    Mat frame;
    string encoded;
    uchar *enc_msg = new uchar[300000];

    while (true) {
        usleep(sleep_time);
        frame = ring_buffer.get();

        if (getCroppedImage) {
            continue;
        }
        else if (quit == true) {
            cout << "[ INFO ] Terminating stream" << endl;
            return;
        }
        else if (!frame.empty()) {
            imencode(".jpg", frame, buf);

            for (int i = 0; i < buf.size(); i++) {
                enc_msg[i] = buf[i];
            }       
            encoded = base64_encode(enc_msg, buf.size());

            try {
                mqtt::delivery_token_ptr pubtok;
                pubtok = client.publish(TOPIC_STREAM, encoded, QOS, false);
            }
            catch (const mqtt::exception& exc) {
                cerr << exc.what() << endl;
                continue;
            }
        }
    }
}

// void frameGrabber_(Mat frame) {

//     do {
//         usleep(10);
//         ring_buffer.put(frame);
//         cout << "\nSize: " << ring_buffer.size();
//     } while (quit != true);

//     return;
// }

int main(int argc, char *argv[]) {

    auto start = chrono::system_clock::now();
    vector<int> reid_memo;
    vector<uchar> buf;
    char str[100];
    int i = 0;

    try {
        
        /** This sample covers 3 certain topologies and cannot be generalized **/
        cout << "InferenceEngine: " << GetInferenceEngineVersion() << endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        mqtt::connect_options connOpts;
        connOpts.set_keep_alive_interval(20);
        connOpts.set_clean_session(true);

        callback cb(client, connOpts);
        client.set_callback(cb);

        cout << "[ MQTT ] Initialized\n";

        cout << "[ MQTT ] Connecting to server " << address << endl;
        client.connect(connOpts, nullptr, cb);
        cout << "[ MQTT ] Waiting for connection" << endl;

        cout << "[ INFO ] Reading input" << endl;
        cout << "[ INFO ] Starting camera" << endl;
        Mat frame = imread(FLAGS_i, IMREAD_COLOR);

        const bool isVideo = frame.empty();
        VideoCapture cap;

        double timestamp = cap.get(CAP_PROP_POS_MSEC);

        if (isVideo && !(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw logic_error("Cannot open input file or camera: " + FLAGS_i);
        }

        const size_t width  = isVideo ? (size_t) cap.get(CAP_PROP_FRAME_WIDTH) : frame.size().width;
        const size_t height = isVideo ? (size_t) cap.get(CAP_PROP_FRAME_HEIGHT) : frame.size().height;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        map<string, InferencePlugin> pluginsForNetworks;
        vector<string> pluginNames = {
                FLAGS_d, FLAGS_d_pa, FLAGS_d_reid
        };

        PersonDetection personDetection;
        PersonReIdentification personReId;

        for (auto && flag : pluginNames) {
            if (flag == "") continue;
            auto i = pluginsForNetworks.find(flag);
            if (i != pluginsForNetworks.end()) {
                continue;
            }
            cout << "[ INFO ] Loading plugin " << flag << endl;
            InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(flag);

            /** Printing plugin version **/
            printPluginVersion(plugin, cout);

            if ((flag.find("CPU") != string::npos)) {
                /** Load default extensions lib for the CPU plugin (e.g. SSD's DetectionOutput)**/
                plugin.AddExtension(make_shared<Extensions::Cpu::CpuExtensions>());
                if (!FLAGS_l.empty()) {
                    // Any user-specified CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    
                    // Uncomment below if used with API 1.4
                    //auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    //plugin.AddExtension(extension_ptr);

                     auto extension_ptr = make_so_pointer<MKLDNNPlugin::IMKLDNNExtension>(FLAGS_l);
                     plugin.AddExtension(static_pointer_cast<IExtension>(extension_ptr));
                }
            }

            if ((flag.find("GPU") != string::npos) && !FLAGS_c.empty()) {
                // Load any user-specified clDNN Extensions
                plugin.SetConfig({ { PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } });
            }
            pluginsForNetworks[flag] = plugin;
            printPluginVersion(plugin, cout);
        }



        /** Per layer metrics **/
        if (FLAGS_pc) {
            for (auto && plugin : pluginsForNetworks) {
                plugin.second.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
            }
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR models and load them to plugins ------------------------------
        Load(personDetection).into(pluginsForNetworks[FLAGS_d]);
        Load(personReId).into(pluginsForNetworks[FLAGS_d_reid]);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Do inference ---------------------------------------------------------
        Blob::Ptr frameBlob;  // Blob to be used to keep processed frame data
        ROI cropRoi;  // cropped image coordinates
        Blob::Ptr roiBlob;  // This blob contains data from cropped image (vehicle or license plate)
        Mat person;  // Mat object containing person data cropped by openCV

        //thread frameGrabber(frameGrabber_, frame);
        thread framePusher(framePusher_);

        int k = 0;

        while(true) {

            while (getCroppedImage) {
                usleep(sleep_time);
                destroyAllWindows();
                cout << "[ INFO ] Standby for Re-id" << endl;

                while (true) {

                    this_thread::sleep_for (std::chrono::seconds(1));

                    if (OK_signal) {
                        cout << "[ INFO ] Performing REID" << endl;

                        reid_enable = true;

                        personReId.enqueue(cropped_image);
                        personReId.submitRequest();
                        personReId.wait();

                        auto reIdVector = personReId.getReidVec();
                        auto foundId = personReId.findMatchingPerson(reIdVector);

                        getCroppedImage = false;
                        OK_signal = false;
                        reid_enable = false;

                        break;
                    }

                    if (close_windows) {
                        cout << "[ UI ] Crop mode exit by UI" << endl;

                        close_windows = false;
                        getCroppedImage = false;
                        break;
                    }
                }
            }

            cout << "[ INFO ] Entering Tagging Mode" << endl;
            
            while(!isSearchMode) {
                /** Start inference & calc performance **/
                usleep(sleep_time);
                typedef chrono::duration<double, ratio<1, 1000>> ms;
                auto total_t0 = chrono::high_resolution_clock::now();               

                do {
                    k++;
                    if (isSearchMode) {
                        break;
                    }

                    if (getCroppedImage) {
                        break;
                    }
                    // get and enqueue the next frame (in case of video)
                    if (isVideo && !cap.read(frame)) {
                        if (frame.empty())
                            break;  // end of video file
                        throw logic_error("Failed to get frame from VideoCapture");
                    }
                    if (FLAGS_auto_resize) {
                        // just wrap Mat object with Blob::Ptr without additional memory allocation
                        frameBlob = wrapMat2Blob(frame);
                        personDetection.setRoiBlob(frameBlob);
                    } else {                          
                        personDetection.enqueue(frame);
                        ring_buffer.put(frame);
                    }

                    // --------------------------- Run Person detection inference --------------------------------------
                    auto t0 = chrono::high_resolution_clock::now();
                    personDetection.submitRequest();
                    personDetection.wait();
                    auto t1 = chrono::high_resolution_clock::now();
                    ms detection = chrono::duration_cast<ms>(t1 - t0);
                    // parse inference results internally (e.g. apply a threshold, etc)
                    personDetection.fetchResults();
                    // -------------------------------------------------------------------------------------------------

                    // --------------------------- Process the results down to the pipeline ----------------------------
                    ms personReIdNetworktime(0);
                    int personReIdInferred = 0;
                    for (auto && result : personDetection.results) {
                        if (result.label == 1) {  // person
                            if (FLAGS_auto_resize) {
                                cropRoi.posX = (result.location.x < 0) ? 0 : result.location.x;
                                cropRoi.posY = (result.location.y < 0) ? 0 : result.location.y;
                                cropRoi.sizeX = min((size_t) result.location.width, width - cropRoi.posX);
                                cropRoi.sizeY = min((size_t) result.location.height, height - cropRoi.posY);
                                roiBlob = make_shared_blob(frameBlob, cropRoi);
                            } else {
                                // To crop ROI manually and allocate required memory (Mat) again
                                auto clippedRect = result.location & Rect(0, 0, width, height);
                                person = frame(clippedRect);                               
                            }
                            string resPersReid = "";

                            //cout << "K-value: " << k << endl;
                            if (personReId.enabled() && (k % 25 == 0)) {

                                if (k == 100) {
                                    k = 0;
                                }

                                for (auto && result : personDetection.results) {
                                    // --------------------------- Run Person Reidentification -----------------------------
                                    if (FLAGS_auto_resize) {
                                        personReId.setRoiBlob(roiBlob);
                                    } else {
                                        personReId.enqueue(person);
                                    }

                                    t0 = chrono::high_resolution_clock::now();
                                    personReId.submitRequest();
                                    personReId.wait();
                                    t1 = chrono::high_resolution_clock::now();

                                    personReIdNetworktime += chrono::duration_cast<ms>(t1 - t0);
                                    personReIdInferred++;

                                    auto reIdVector = personReId.getReidVec();

                                    /* Check cosine similarity with all previously detected persons.
                                       If it's new person it is added to the global Reid vector and
                                       new global ID is assigned to the person. Otherwise, ID of
                                       matched person is assigned to it. */
                                    auto foundId = personReId.findMatchingPerson(reIdVector);

                                    resPersReid = "REID: " + to_string(foundId);

                                    for (int i=0; i <= foundId; i++) {

                                        if (reid_memo.empty()) {
                                            reid_memo.push_back(foundId);
                                            continue;
                                        }
                                        else if ((reid_memo.size() - 1) == foundId) {
                                            
                                            imencode(".jpg", person, buf);
                                            imageDB.push_back(buf);


                                            cout << "[ INFO ] Person database size: " << imageDB.size() << endl;

                                            uchar *enc_msg = new uchar[buf.size()];

                                            for (int i = 0; i < buf.size(); i++) {
                                                enc_msg[i] = buf[i];
                                            }       
                                            string encoded = base64_encode(enc_msg, buf.size());

                                            try {

                                                //cout << "[ MQTT ] Sending image to server ";
                                                //mqtt::delivery_token_ptr pubtok;
                                                //pubtok = client.publish(TOPIC_IMAGE, encoded, QOS, false);
                                                //cout << "with token ID " << pubtok->get_message_id() << " for size of "
                                                //    << pubtok->get_message()->get_payload().size() << " bytes\n";
                                                //pubtok->wait_for(TIMEOUT);
                                            }
                                            catch (const mqtt::exception& exc) {
                                                cerr << exc.what() << endl;
                                                continue;
                                            }

                                            //cout << "size " << reid_memo.size() << " = " << foundId << endl;
                                            reid_memo.push_back(foundId);
                                        }
                                        else {
                                            continue;
                                        }
                                    }
                                }
                            }

                            //--------------------------- Process outputs -----------------------------------------
                            if (!resPersReid.empty()) {
                                putText(frame,
                                            resPersReid,
                                            Point2f(result.location.x, result.location.y + 30),
                                            FONT_HERSHEY_COMPLEX_SMALL,
                                            0.7,
                                            Scalar(255, 255, 255));
                                if (FLAGS_r) {
                                    cout << "Person Reidentification results:" << resPersReid << endl;
                                }
                            }
                            rectangle(frame, result.location, Scalar(0, 255, 0), 2);
                        }
                    }

                    // --------------------------- Execution statistics ------------------------------------------------
                    ostringstream out;
                    out << "Person detection time  : " << fixed << setprecision(2) << detection.count()
                        << " ms ("
                        << 1000.f / detection.count() << " fps)";
                    putText(frame, out.str(), Point2f(10, 40), FONT_HERSHEY_TRIPLEX, 0.75, Scalar(0, 0, 255));

                    if (personDetection.results.size()) {
                        if (personReId.enabled() && personReIdInferred) {
                            float average_time = personReIdNetworktime.count() / personReIdInferred;
                            out.str("");
                            out << "Person Reidentification time (averaged over " << personReIdInferred
                                << " detections) :" << fixed << setprecision(2) << average_time
                                << " ms " << "(" << 1000.f / average_time << " fps)";
                            putText(frame, out.str(), Point2f(10, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255));
                            if (FLAGS_r) {
                                cout << out.str() << endl;;
                            }
                        }
                    }

                    if (!FLAGS_no_show) {
                        //namedWindow("Detection results", WINDOW_NORMAL);
                        //resizeWindow("Detection results", 848,480);
                        imshow("Detection results", frame);
                    }
                    // for still images wait until any key is pressed, for video 1 ms is enough per frame
                    const int key = waitKey(isVideo ? 1 : 0);

                    // Esc
                    if (27 == key) {
                        quit = true;
                        framePusher.join();
                        //frameGrabber.join();
                        break;
                    }

                } while (isVideo);

                if (quit == true) {
                    break;
                }
                if (getCroppedImage == true) {
                    break;
                }
            }

            if (quit == true) {
                break;
            }

            cout << "[ INFO ] Entering Search Mode" << endl;

            while (isSearchMode) {
                /** Start inference & calc performance **/
                usleep(sleep_time);
                typedef chrono::duration<double, ratio<1, 1000>> ms;
                auto total_t0 = chrono::high_resolution_clock::now();
                
                do {

                    if (!isSearchMode) {
                        break;
                    }
                    if (getCroppedImage) {
                        break;
                    }

                    try {
                        // get and enqueue the next frame (in case of video)
                        if (isVideo && !cap.read(frame)) {
                            if (frame.empty())
                                break;  // end of video file
                            throw logic_error("Failed to get frame from VideoCapture");
                        }
                        if (FLAGS_auto_resize) {
                            // just wrap Mat object with Blob::Ptr without additional memory allocation
                            frameBlob = wrapMat2Blob(frame);
                            personDetection.setRoiBlob(frameBlob);
                        } else {
                            personDetection.enqueue(frame);
                        }
                        // --------------------------- Run Person detection inference --------------------------------------
                        auto t0 = chrono::high_resolution_clock::now();
                        personDetection.submitRequest();
                        personDetection.wait();
                        auto t1 = chrono::high_resolution_clock::now();
                        ms detection = chrono::duration_cast<ms>(t1 - t0);
                        // parse inference results internally (e.g. apply a threshold, etc)
                        personDetection.fetchResults();
                        // -------------------------------------------------------------------------------------------------

                        // --------------------------- Process the results down to the pipeline ----------------------------
                        ms personReIdNetworktime(0);
                        int personReIdInferred = 0;
                        for (auto && result : personDetection.results) {
                            if (result.label == 1) {  // person
                                if (FLAGS_auto_resize) {
                                    cropRoi.posX = (result.location.x < 0) ? 0 : result.location.x;
                                    cropRoi.posY = (result.location.y < 0) ? 0 : result.location.y;
                                    cropRoi.sizeX = min((size_t) result.location.width, width - cropRoi.posX);
                                    cropRoi.sizeY = min((size_t) result.location.height, height - cropRoi.posY);
                                    roiBlob = make_shared_blob(frameBlob, cropRoi);
                                } else {
                                    // To crop ROI manually and allocate required memory (Mat) again
                                    auto clippedRect = result.location & Rect(0, 0, width, height);
                                    person = frame(clippedRect);                            
                                }
                                string resPersReid = "";

                                if (personReId.enabled()) {
                                    // --------------------------- Run Person Reidentification -----------------------------
                                    if (FLAGS_auto_resize) {
                                        personReId.setRoiBlob(roiBlob);
                                    } else {
                                        personReId.enqueue(person);
                                    }

                                    t0 = chrono::high_resolution_clock::now();
                                    personReId.submitRequest();
                                    personReId.wait();
                                    t1 = chrono::high_resolution_clock::now();

                                    personReIdNetworktime += chrono::duration_cast<ms>(t1 - t0);
                                    personReIdInferred++;

                                    auto reIdVector = personReId.getReidVec();
                                    auto foundId = personReId.findMatchingPerson(reIdVector);

                                    resPersReid = "TARGET";
                                }

                                //--------------------------- Process outputs -----------------------------------------
                                if (!resPersReid.empty() && targetQueryLocated == true) {
                                    putText(frame,
                                                resPersReid,
                                                Point2f(result.location.x, result.location.y + 30),
                                                FONT_HERSHEY_COMPLEX_SMALL,
                                                0.7,
                                                Scalar(255, 255, 255));
                                    if (FLAGS_r) {
                                        cout << "Person Reidentification results:" << resPersReid << endl;
                                    }
                                }
                                if (targetQueryLocated == true) {
                                    rectangle(frame, result.location, Scalar(0, 255, 0), 2);
                                }
                            }
                        }

                        // --------------------------- Execution statistics ------------------------------------------------
                        ostringstream out;
                        out << "Person detection time  : " << fixed << setprecision(2) << detection.count()
                            << " ms ("
                            << 1000.f / detection.count() << " fps)";
                        putText(frame, out.str(), Point2f(10, 40), FONT_HERSHEY_TRIPLEX, 0.75, Scalar(0, 0, 255));

                        if (personDetection.results.size()) {
                            if (personReId.enabled() && personReIdInferred) {
                                float average_time = personReIdNetworktime.count() / personReIdInferred;
                                out.str("");
                                out << "Person Reidentification time (averaged over " << personReIdInferred
                                    << " detections) :" << fixed << setprecision(2) << average_time
                                    << " ms " << "(" << 1000.f / average_time << " fps)";
                                putText(frame, out.str(), Point2f(10, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255));
                                if (FLAGS_r) {
                                    cout << out.str() << endl;;
                                }
                            }
                        }

                        if (!FLAGS_no_show) {
                            //namedWindow("Detection results", WINDOW_NORMAL);
                            //resizeWindow("Detection results", 1280,720);
                            imshow("Detection results", frame);
                        }
                        // for still images wait until any key is pressed, for video 1 ms is enough per frame
                        const int key = waitKey(isVideo ? 1 : 0);

                        if (27 == key)  // 
                            quit = true;
                            framePusher.join();
                            break;
                    }
                    catch (exception& e) {
                        cerr << "[ ERROR ] Exception at SearchMode " << e.what() << endl; 
                    }
                } while (isVideo);

                if (quit == true) {
                    break;
                }
            }

            if (quit == true) {
                break;
            }
        }

        auto toks = client.get_pending_delivery_tokens();

        if (!toks.empty()) {
            cout << "[ ERROR ] There are pending delivery tokens!\n";
        }
        client.disconnect();

        auto total_t1 = chrono::high_resolution_clock::now();
        //ms total = chrono::duration_cast<ms>(total_t1 - total_t0);

        //cout << "[ INFO ] Inference uptime: " << total.count()/1000 << " seconds" << endl;

        /** Show performace results **/
        if (FLAGS_pc) {
            for (auto && plugin : pluginsForNetworks) {
                cout << "[ INFO ] Performance counts for " << plugin.first << " plugin";
                printPerformanceCountsPlugin(plugin.second, cout);
            }
        }
    }
    catch (const exception& error) {
        cerr << "[ ERROR ] " << error.what() << endl;
    }
    catch (...) {
        cerr << "[ ERROR ] Unknown/internal exception happened." << endl;
        return 1;
    }

    // Disconnect
    destroyAllWindows();
    cout << "[ INFO ] Disconnected from server successfully" << endl;
    cout << "[ INFO ] Program ended successfully" << endl;

    return 0;
}
