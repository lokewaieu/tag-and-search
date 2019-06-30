/*Server v1.0 22-1-2019
         v1.1 27-1-2019 - Added Smart UI
         v1.2 28-1-2019 - Added Top 3 cosSim match function
         v1.3 30-1-2019 - Fix some boolean logic errors during UI navigation*/

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <inference_engine.hpp>
//#include <samples/ocv_common.hpp>
#include <samples/common.hpp>
#include "crossroad_camera_sample.hpp"
#include <ext_list.hpp>

#include "mqtt/async_client.h"
#include "base64.h"

#define ROOT "/home/ubuntu/Desktop/gallery/"

using namespace cv;
using namespace std;
using namespace InferenceEngine;

static const string base64_chars = 
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

const string DFLT_SERVER_ADDRESS    {"192.168.1.102:1883"};
const string DFLT_CLIENT_ID         {"Server"};
const string TOPIC_IMAGE_1          {"Camera_1"};
const string TOPIC_IMAGE_2          {"Camera_2"};
const string TOPIC_FEATURE_1        {"Feature_Camera_1"};
const string TOPIC_FEATURE_2        {"Feature_Camera_2"};
const string TOPIC_SEARCH_START     {"search_mode_start"};
const string TOPIC_SEARCH_STOP      {"search_mode_stop"};
const string TOPIC_QUERY_VECTOR     {"query_vector"};
const string TOPIC_QUERY_IMAGE      {"query_image"};            // original frame after pausing video
const string TOPIC_UI               {"display_ui"};             // smartUI main frame display
const string TOPIC_TOP_1            {"top_first"};              // best reid match cropped image
const string TOPIC_TOP_2            {"top_second"};             // second best match
const string TOPIC_TOP_3            {"top_third"};              // third best match
const string TOPIC_MOUSE_COOR       {"mouse_coor"};             // mouse coordinates
const string TOPIC_DONE_DROP        {"done_crop"};              // click to crop
const string TOPIC_CROPPED_IMAGE    {"cropped_image"};          // cropped image display
const string TOPIC_REID_RESULTS     {"reid_results"};           
const string TOPIC_OK_SIGNAL        {"ok_signal"};              // start reid confirmation
const string TOPIC_CLOSE_WINDOW     {"close_window"};

const char* LWT_PAYLOAD = "\nI'm dying unexpectedly. Bye :(\n";

const int QOS = 1;
const int  N_RETRY_ATTEMPTS = 999;
const auto TIMEOUT = chrono::seconds(5);

vector<vector<float>> universalReIdVec;
vector<vector<uchar>> imageDB;
vector<float> tempQueryVector;
float top_3_score[3];

Mat query_image;
int mousX, mousY;
bool getQueryImage = false;
bool done_crop = false;
bool OK_signal = false;
bool startSearch = false;
bool startTag = false;
bool endProgram = false;
bool close_window = false;
bool top_1 = false;
bool top_2 = false;
bool top_3 = false;

string  address     = DFLT_SERVER_ADDRESS,
        clientID    = DFLT_CLIENT_ID;

mqtt::async_client client(address, clientID);

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
        cout << "[ MQTT ] Connection success with QoS " << QOS << endl;
        // cout << "[ MQTT ] Subscribing to topic " << TOPIC_IMAGE
        //  << " for client " << DFLT_CLIENT_ID
        //  << " using QoS " << QOS << endl
        //  << "\nPress Q<Enter> to quit\n" << endl;

        cli_.subscribe(TOPIC_IMAGE_1, QOS, nullptr, subListener_);
        cli_.subscribe(TOPIC_FEATURE_1, QOS, nullptr, subListener_);
        cli_.subscribe(TOPIC_IMAGE_2, QOS, nullptr, subListener_);
        cli_.subscribe(TOPIC_FEATURE_2, QOS, nullptr, subListener_);
        cli_.subscribe(TOPIC_QUERY_VECTOR, QOS, nullptr, subListener_);
        cli_.subscribe(TOPIC_QUERY_IMAGE, QOS, nullptr, subListener_);
        cli_.subscribe(TOPIC_MOUSE_COOR, QOS, nullptr, subListener_);
        cli_.subscribe(TOPIC_DONE_DROP, QOS, nullptr, subListener_);
        cli_.subscribe(TOPIC_OK_SIGNAL, QOS, nullptr, subListener_);
        cli_.subscribe(TOPIC_CLOSE_WINDOW, QOS, nullptr, subListener_);
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
        //cout << "Message arrived" << endl;
        //cout << "\ttopic: '" << msg->get_topic() << "'" << endl;
        //cout << "\tpayload: '" << msg->to_string() << "'\n" << endl;

        if (msg->get_topic() == "Feature_Camera_1") {
            cout << "[ MQTT ] Receiving feature vector from camera 1\n";

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
            // ofstream output_file("/home/loke/Desktop/acertainvector.csv");
            // ostream_iterator<float> output_iterator(output_file, "\n");
            // copy(decodedVec_asFloat.begin(), decodedVec_asFloat.end(), output_iterator);


            // for (auto i = decodedVec_asFloat.begin(); i != decodedVec_asFloat.end(); i++)
            //  cout << "\n" << *i << " ";

            universalReIdVec.push_back(decodedVec_asFloat);
            tempQueryVector = decodedVec_asFloat;
        }
        if (msg->get_topic() == "Feature_Camera_2") {
            cout << "[ MQTT ] Receiving feature vector from camera 2\n";

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

            universalReIdVec.push_back(decodedVec_asFloat);
            tempQueryVector = decodedVec_asFloat;
        }
        if (msg->get_topic() == "Camera_1") {
            cout << "[ MQTT ] Receiving cropped image from camera 1\n";

            string image_asString = msg->to_string();

            image_asString = base64_decode(image_asString);

            vector<uchar> data(image_asString.begin(), image_asString.end());

            if (!data.empty()) {

                imageDB.push_back(data);
            }
        }
        if (msg->get_topic() == "Camera_2") {
            cout << "[ MQTT ] Receiving cropped image from camera 2\n";

            string image_asString = msg->to_string();

            image_asString = base64_decode(image_asString);

            vector<uchar> data(image_asString.begin(), image_asString.end());

            if (!data.empty()) {

                imageDB.push_back(data);
            }
        }
        if (msg->get_topic() == "query_image") {
            cout << "[ MQTT ] Query image received\n";

            string image_asString = msg->to_string();

            image_asString = base64_decode(image_asString);

            vector<uchar> data(image_asString.begin(), image_asString.end());
            query_image = imdecode(data, 1);

            getQueryImage = true;
        }
        if (msg->get_topic() == "mouse_coor") {
            // cout << "[ MQTT ] Mouse input received\n";

            string coor_asString = msg-> to_string();

            istringstream ss(coor_asString);

            if (ss >> mousX >> mousY) {
                //cout << "(" << mousX << ", " << mousY << ")" << endl;
            }
        }
        if (msg->get_topic() == "close_window") {
            cout << "[ MQTT ] Window closed\n";

            close_window = true;
        }
        if (msg->get_topic() == "done_crop") {

            done_crop = true;
        }
        if (msg->get_topic() == "ok_signal") {
            cout << "[ MQTT ] OK\n";

            OK_signal = true;
        }
    }

    void delivery_complete(mqtt::delivery_token_ptr token) override {}

public:
    callback(mqtt::async_client& cli, mqtt::connect_options& connOpts)
                : nretry_(0), cli_(cli), connOpts_(connOpts), subListener_("Subscription") {}
};

/////////////////////////////////////////////////////////////////////////////

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

void find_top_3(float arr[], int arr_size, double FLAGS_t_reid) { 
    
    int i;
    float first, second, third; 
   
    /* There should be atleast two elements */
    if (arr_size < 3) 
    { 
        cout << " Invalid Input. Size of  " << arr_size << endl; 
        return; 
    } 
   
    third = first = second = -2147483648; 
    for (i = 0; i < arr_size; i++) 
    { 
        /* If current element is smaller than first*/
        if (arr[i] > first) 
        { 
            third = second; 
            second = first; 
            first = arr[i];
            
            top_3_score[0] = i;
        } 
   
        /* If arr[i] is in between first and second then update second  */
        else if (arr[i] > second) 
        { 
            third = second; 
            second = arr[i];
            
            top_3_score[1] = i; 
        } 
   
        else if (arr[i] > third) 
            third = arr[i];
            
            top_3_score[2] = i; 
    }
    
    for (i = 0; i < arr_size; i++) {

        if (arr[i] == first) {
            top_3_score[0] = i;

            if (arr[i] > FLAGS_t_reid) {
                top_1 = true;
            }
        }
        if (arr[i] == second) {
            top_3_score[1] = i;

            if (arr[i] > FLAGS_t_reid) {
                top_2 = true;
            }
        }
        if (arr[i] == third) {
            top_3_score[2] = i;

            if (arr[i] > FLAGS_t_reid) {
                top_3 = true;
            }
        }   
    }

    cout << "3 largest: " << first << "," << second << "," << third << endl;
} 

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
    mutable bool _enabled = true;

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

    PersonReIdentification() : BaseDetection(FLAGS_m_reid, "Person Reidentification Retail") {}

    unsigned long int findMatchingPerson(const vector<float> &newReIdVec) {
        float cosSim;
        auto size = universalReIdVec.size();
        float cosSimArr[size];
        vector<uchar> buf;

        /* assigned REID is index of the matched vector from the globalReIdVec */
        for (auto i = 0; i < size; ++i) {

            cosSim = cosineSimilarity(newReIdVec, universalReIdVec[i]);
            cout << "[ REID ] cosineSimilarity: " << cosSim << endl;

            if (FLAGS_r) {
                cout << "cosineSimilarity: " << cosSim << endl;
            }
            if (cosSim > FLAGS_t_reid) {
                /* We substitute previous person's vector by a new one characterising
                 * last person's position */

                if (getQueryImage) {
                    //Do not update vector feature
                    cosSimArr[i] = cosSim;
                }
                else {
                    universalReIdVec[i] = newReIdVec;
                }
            }
        }

        if (getQueryImage) {

            cout << " Size of  " << size << endl;
            
            if (size >= 3) {
                cout << "Find top 3" << endl;
                find_top_3(cosSimArr, size, FLAGS_t_reid);
                int size_n = sizeof(top_3_score)/sizeof(top_3_score[0]);

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
                            cout << "[ MQTT ] Sending potential match to UI" << endl;
                            mqtt::delivery_token_ptr pubtok;
                            pubtok = client.publish(TOPIC_TOP_1, encoded, QOS, false);
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
                            cout << "[ MQTT ] Sending potential match to UI" << endl;
                            mqtt::delivery_token_ptr pubtok;
                            pubtok = client.publish(TOPIC_TOP_2, encoded, QOS, false);
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
                            cout << "[ MQTT ] Sending potential match to UI" << endl;
                            mqtt::delivery_token_ptr pubtok;
                            pubtok = client.publish(TOPIC_TOP_3, encoded, QOS, false);
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

        //universalReIdVec.push_back(newReIdVec);

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

int main(int argc, char* argv[]) 
{
    char keyIn;
    vector<uchar> buf;

    mqtt::connect_options connOpts;
    connOpts.set_keep_alive_interval(20);
    connOpts.set_clean_session(true);

    callback cb(client, connOpts);
    client.set_callback(cb);

    try {
        cout << "Connecting to MQTT server " << DFLT_SERVER_ADDRESS << endl;
        client.connect(connOpts, nullptr, cb);

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        map<string, InferencePlugin> pluginsForNetworks;
        vector<string> pluginNames = {
                FLAGS_d, FLAGS_d_reid
        };

        PersonReIdentification personReId;
        PersonDetection personDetection;

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
                    // auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    // plugin.AddExtension(extension_ptr);

                    auto extension_ptr = make_so_pointer<MKLDNNPlugin::IMKLDNNExtension>(FLAGS_l);
                    plugin.AddExtension(static_pointer_cast<IExtension>(extension_ptr));
                }
            }

            if ((flag.find("GPU") != string::npos) && !FLAGS_c.empty()) {
                // Load any user-specified clDNN Extensions
                plugin.SetConfig({ { PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } });
            }
            pluginsForNetworks[flag] = plugin;
        }

        // --------------------------- 2. Read IR models and load them to plugins ------------------------------
        Load(personDetection).into(pluginsForNetworks[FLAGS_d]);
        Load(personReId).into(pluginsForNetworks[FLAGS_d_reid]);
        cout << "[ INFO ] Initialization complete\nQ to start Re-id | C to quit from Re-id interface\n";


        // --------------------------- 3. Do inference ---------------------------------------------------------
        while(true) {           
            try {
                // Crop-Query Mode
                if (getQueryImage) {
                    Mat dst;
                    Mat person;
                    Mat bbox_frame_checker;

                    while(query_image.empty()) {
                        cout << "[ WARNING ] Empty Query Image!" << endl;
                        continue;
                    }

                    //Mat query_image = imread("/home/ubuntu/Documents/tes.jpg");
                    personDetection.enqueue(query_image);
                    personDetection.submitRequest();
                    personDetection.wait();
                    personDetection.fetchResults();

                    vector<int> coordinatesX;
                    vector<int> coordinatesY;
                    vector<int> coordinatesVerX;
                    vector<int> coordinatesVerY;

                    for (auto && result : personDetection.results) {
                        if (result.label == 1) {

                            //rectangle(query_image, result.location, Scalar(0, 255, 0), 2);
                            //circle(query_image, Point(result.location.x, result.location.y), 50, Scalar(0, 255 ,0), 2);
                            // imshow("frame", query_image);
                            // waitKey(0);

                            cout << "Location x: " << result.location.x << endl;
                            cout << "Location y: " << result.location.y << endl;
                            cout << "Location width" << result.location.width << endl;
                            cout << "Location height" << result.location.height << endl;

                            coordinatesX.push_back(result.location.x);
                            coordinatesY.push_back(result.location.y);
                            coordinatesVerX.push_back(result.location.width);
                            coordinatesVerY.push_back(result.location.height);
                        }                       
                    }

                    Mat bbox_frame = query_image.clone();

                    while (true) {

                        for (int i = 0; i < coordinatesX.size(); i++) {

                            if (mousX > coordinatesX[i] && mousX < (coordinatesX[i] + coordinatesVerX[i]) 
                                && mousY > coordinatesY[i] && mousY < (coordinatesY[i] + coordinatesVerY[i])) {

                                    Point A(coordinatesX[i], coordinatesY[i]);
                                    Point B(coordinatesX[i] + coordinatesVerX[i], coordinatesY[i] + coordinatesVerY[i]);
                                    Rect boundingBox(A, B);

                                    //cout << boundingBox << endl;

                                    //cout << "BOX: " << boundingBox << endl;

                                    rectangle(bbox_frame, boundingBox, Scalar(0, 255, 0), 2);

                                    if (bbox_frame_checker.empty()) {
                                        imencode(".jpg", bbox_frame, buf);
                                        uchar *enc_msg = new uchar[buf.size()];

                                        for (int i = 0; i < buf.size(); i++) {
                                            enc_msg[i] = buf[i];
                                        }
                                        string encoded = base64_encode(enc_msg, buf.size());

                                        try {
                                            cout << "[ MQTT ] Sending updated bbox image to UI" << endl;
                                            mqtt::delivery_token_ptr pubtok;
                                            pubtok = client.publish(TOPIC_UI, encoded, QOS, false);
                                            cout << " with token ID " << pubtok->get_message_id() << " for size of "
                                                << pubtok->get_message()->get_payload().size() << " bytes" << endl;
                                        }
                                        catch (const mqtt::exception& exc) {
                                            cerr << exc.what() << endl;
                                            continue;
                                        }

                                        bbox_frame_checker = bbox_frame.clone();
                                        bbox_frame = query_image.clone();                                       
                                    }                                   
                                    else {
                                        Mat bbox_frame_gray;
                                        Mat bbox_frame_checker_gray;

                                        cvtColor(bbox_frame, bbox_frame_gray, COLOR_BGR2GRAY);
                                        cvtColor(bbox_frame_checker, bbox_frame_checker_gray, COLOR_BGR2GRAY);

                                        bitwise_xor(bbox_frame_gray, bbox_frame_checker_gray, dst);
                                        if (countNonZero(dst) > 0) {
                                            imencode(".jpg", bbox_frame, buf);
                                            uchar *enc_msg = new uchar[buf.size()];

                                            for (int i = 0; i < buf.size(); i++) {
                                                enc_msg[i] = buf[i];
                                            }
                                            string encoded = base64_encode(enc_msg, buf.size());

                                            try {
                                                cout << "[ MQTT ] Sending updated bbox image to UI" << endl;
                                                mqtt::delivery_token_ptr pubtok;
                                                pubtok = client.publish(TOPIC_UI, encoded, QOS, false);
                                                cout << " with token ID " << pubtok->get_message_id() << " for size of "
                                                    << pubtok->get_message()->get_payload().size() << " bytes" << endl;
                                            }
                                            catch (const mqtt::exception& exc) {
                                                cerr << exc.what() << endl;
                                                continue;
                                            }
                                            
                                            bbox_frame_checker = bbox_frame.clone();
                                            bbox_frame = query_image.clone();
                                        }
                                    }

                                    if (done_crop) {

                                        cout << "[ MQTT ] Selection confirmed from UI\n";

                                        auto clippedRect = boundingBox & Rect(0, 0, query_image.size().width, query_image.size().height);
                                        person = query_image(clippedRect);

                                        imencode(".jpg", person, buf);
                                        uchar *enc_msg = new uchar[buf.size()];

                                        for (int i = 0; i < buf.size(); i++) {
                                            enc_msg[i] = buf[i];
                                        }
                                        string encoded = base64_encode(enc_msg, buf.size());

                                        try {
                                            cout << "[ MQTT ] Sending cropped image to UI" << endl;
                                            mqtt::delivery_token_ptr pubtok;
                                            pubtok = client.publish(TOPIC_CROPPED_IMAGE, encoded, QOS, false);
                                            cout << " with token ID " << pubtok->get_message_id() << " for size of "
                                                << pubtok->get_message()->get_payload().size() << " bytes" << endl;
                                        }
                                        catch (const mqtt::exception& exc) {
                                            cerr << exc.what() << endl;
                                            continue;
                                        }

                                        done_crop = false;
                                    }

                                    if (close_window) {

                                        cout << "[ UI ] Closing Window1" << endl; 

                                        coordinatesX.clear();
                                        coordinatesY.clear();
                                        coordinatesVerX.clear();
                                        coordinatesVerY.clear();                                      

                                        break;
                                    }

                                    if (OK_signal) {

                                        cout << "[ UI ] OK1" << endl;

                                        coordinatesX.clear();
                                        coordinatesY.clear();
                                        coordinatesVerX.clear();
                                        coordinatesVerY.clear();

                                        break;
                                    }
                            }
                        }

                        if (close_window) {

                            cout << "[ UI ] Closing Window2" << endl; 

                            coordinatesX.clear();
                            coordinatesY.clear();
                            coordinatesVerX.clear();
                            coordinatesVerY.clear();                                      

                            close_window = false;
                            getQueryImage = false;
                            break;
                        }

                        if (OK_signal) {

                            cout << "[ UI ] OK2" << endl;

                            coordinatesX.clear();
                            coordinatesY.clear();
                            coordinatesVerX.clear();
                            coordinatesVerY.clear();

                            getQueryImage = false;
                            OK_signal = false;

                            break;
                        }
                    }
                }
                if (startSearch) {
                    cout << "[ STARTING SEARCH MODE ]" << endl;
                    mqtt::delivery_token_ptr pubtok;
                    pubtok = client.publish(TOPIC_SEARCH_START, "start", QOS, false);

                    // cout << "[ MQTT ] Sending image to server ";
                    // mqtt::delivery_token_ptr pubtok;
                    // pubtok = client.publish(TOPIC_IMAGE, encoded, QOS, false);
                    // cout << "with token ID " << pubtok->get_message_id() << " for size of "
                    //  << pubtok->get_message()->get_payload().size() << " bytes\n";

                    stringstream ss1;
                    if (tempQueryVector.size() == 0) {
                        cout << "[ ERROR ] Empty query vector. Aborting search mode" << endl;
                    }
                    else {

                        for (size_t i = 0; i < tempQueryVector.size(); i++) {
                            ss1 << "\n";
                            ss1 << tempQueryVector[i];
                        }

                        string s = ss1.str();

                        cout << "[ MQTT ] Sending query vector to all cameras ";
                        pubtok = client.publish(TOPIC_QUERY_VECTOR, s, QOS, false);
                        cout << "with token ID " << pubtok->get_message_id() << " for size of "
                            << pubtok->get_message()->get_payload().size() << " bytes\n";
                    }
                    startSearch = false;                    
                }
                if (startTag) {
                    cout << "[ TAGGING MODE ]" << endl;
                    mqtt::delivery_token_ptr pubtok;
                    pubtok = client.publish(TOPIC_SEARCH_STOP, "stop", QOS, false);
                    startTag = false;
                }
                if (endProgram) {
                    break;
                }
            }
            catch (const exception& error) {
                cerr << "[ ERROR ] " << error.what() << endl;
            }
        }
    }
    catch (const mqtt::exception&) {
        cerr << "[ ERROR ] Unable to connect to MQTT server: " << DFLT_SERVER_ADDRESS << endl;
        return 1;
    }
    catch (const exception& error) {
        cerr << "[ ERROR ] " << error.what() << endl;
    }

    try {
        cout << "[ INFO ] Disconnecting from MQTT Server" << endl;
        client.disconnect()->wait();
    }
    catch (const mqtt::exception& exc) {
        cerr << exc.what() << endl;
        return 1;
    }

    return 0;
}