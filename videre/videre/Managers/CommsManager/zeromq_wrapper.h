/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: March. 2016
 *
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *******************************************************************/

#ifndef ZEROMQ_WRAPPER
#define ZEROMQ_WRAPPER

#include <memory>
#include <string>
#include <exception>

#include <opencv2/highgui.hpp>
//#include <opencv2/imgcodecs.hpp>

/* We use zeromq for message transportation */
#include <zmq.hpp>
#include "../../Utils/logger.h"


namespace videre
{

/**
 * Simple communication using 0MQ
 * The class has a method for responding to messages sent from HOPS
 * Framework and a method for publishing results to HOPS Framework (or any
 * other subscribers).
 */
    class ZeroMQWrapper
    {

    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::unique_ptr<zmq::context_t> zcontext_sptr_;

        /* Command/Response socket... commands to control the
         * Vision system are send to the Vision System over this
         * socket and the Vision system responds/replies to each
         * command over this socket.*/
        std::unique_ptr<zmq::socket_t> zsocket_cmdresponse_sptr_;

        /* publish compute vision results... such as target
         * locations, GPS Denied vehical estimated location information
         * and such.*/
        std::unique_ptr<zmq::socket_t> zsocket_pub_visionresults_sptr_;

        /* publish video to ground*/
        std::unique_ptr<zmq::socket_t> zsocket_pub_videoout_sptr_;

        /* Publish monitor data... This data/information is for
         * monitoring the operation of the Vision system.*/
        std::unique_ptr<zmq::socket_t> zsocket_pub_visionsysmonitor_sptr_;

        /* subscription to telemetry socket */
        std::unique_ptr<zmq::socket_t> zsocket_sub_telemeteryinput_sptr_;

        /* Publish monitor data... This data/information is for
        * sending data and information to the Nexus BCI (Brain Computer Interface.*/
        std::unique_ptr<zmq::socket_t> zsocket_pub_nexus_bci_sptr_;

        /* subscription to telemetry socket This data/information is for
        * receiving data and information from the Nexus BCI (Brain Computer Interface.*/
        std::unique_ptr<zmq::socket_t> zsocket_sub_nexus_bci_sptr_;

        zmq::message_t _reply_zmqmsg;
        zmq::message_t _publish_visionresults_zmqmsg;
        zmq::message_t _publish_image_zmgmsg;
        zmq::message_t _publish_monitor_zmqmsg;
        zmq::message_t subscribe_telemetry_zmqmsg_sptr_;
        zmq::message_t _publish_nexus_bci_zmqmsg;

        bool zmq_initialized_;

        /* set this to true with accessor if we are doing unit tests */
        bool testing_without_socket_;
    public:
        /**
         * Constructor for HOPSComm
         */
        ZeroMQWrapper();

        /**
         * Initialize using a host name for the response message and a host name
         * for the publish messages.
         * @param host_rep host name for responses.
         * @param host_pub host name for published data.
         * @param host_pub_video host name for published video.
         * @param host_sub_tel host name for subscribed telemetry.
         * @param host_pub_mon host name for published monitor data.
         * @throws VisionIPCException if there is trouble connecting to 0MQ sockets.
         */
        void Initialize(std::string host_rep,
                        std::string host_pub,
                        std::string host_pub_video,
                        std::string host_sub_tel,
                        std::string host_pub_mon,
                        std::string host_pub_nexus_bci_port,
                        std::string host_sub_nexus_bci_port);                        

        /**
         * Receive a request from HOPS Framework, but don't wait.
         * @param req_ptr is a pointer to the message_t object
         * @return true if a message has been received.
         */
        bool ReceiveRequestFromHOPS(zmq::message_t *req_ptr);

        /**
         * Respond to HOPS Framework
         * @param response string to send back to HOPS Framework.
         * @return true if response was sent.
         * @throws VisionIPCException if HOPSComm has not been initialized.
         */
        bool SendResponse(std::string response);

        /**
         * Publish results to HOPS Framework or to whoever is listening.
         * @param publish string containing published data.
         * @return true if published data was sent.
         * @throws VisionIPCException if HOPSComm has not been initialized.
         */
        bool PublishVisionProcResults(std::string publish);

        /**
         * Publish video to whoever is listening.
         * @param publish_video buffer contains video.
         * @return true if published data was sent.
         * @throws VisionIPCException if HOPSComm has not been initialized.
         */
        bool PublishImageMessage(std::string imgMsg);

        /**
         * Publish monitor data to whoever is listening.
         * @param json_data string of data containing monitor info.
         * @return true if published data was sent.
         * @throws VisionIPCException if HOPSComm has not been initialized.
         */
        bool PublishMonitor(std::string json_data);

        /**
         * Check for telemetry data from HOPS. This is a subscription, no
         * response is necessary. This routine will make sure that we have the
         * latest information because it polls the buffer until all messages
         * have been read. (Note, in future versions, use ZMQ_CONFLATE.)
         * @param sub_ptr is the message that has been received and is sent to protobuf.
         * @return true if a message has been received.
         */
        bool SubscribedTelemetryFromHOPS(zmq::message_t *sub_ptr);

        bool PublishNexusBCI(uint8_t *dataBuf, int dataLength);

        bool ReceiveNexusBCIMessage(zmq::message_t *sub_ptr);

        /**
         * Call this when you are done.
         */
        void Close();

    private:
        /**
         * Helper routine that sends nothing if we are in test mode.
         * Test mode is on when testing_without_socket_ = true;
         * @return true if data has been sent.
         */
        bool SendOnSocketRep();

        /**
         * Helper routine that sends nothing if we are in test mode.
         * Test mode is on when testing_without_socket_ = true;
         * @return true if data has been sent.
         */
        bool SendOnSocketPubVisionProcResults();

        /**
         * Helper routine that sends nothing if we are in test mode.
         * Test mode is on when testing_without_socket_ = true;
         * @return true if data has been sent.
         */
        bool SendOnSocketPubVideo();

        /**
         * Helper routine that sends nothing if we are in test mode.
         * Test mode is on when testing_without_socket_ = true;
         * @return true if data has been sent.
         */
        bool SendOnSocketPubMonitor();

        bool SendOnSocketPubNexusBCI();

    public:

        /**
         * Testing without sockets means that we wont be trying to send anything.
         * This mode is useful for unit testing where we don't want to send anything
         * on the sockets.
         * @return true if we are unit testing.
         */
        bool testing_without_socket() const
        {
            return testing_without_socket_;
        }

        /**
         * Testing without sockets means that we wont be trying to send anything.
         * This mode is useful for unit testing where we don't want to send anything
         * on the sockets.
         * @param value should be set to true if we want to do unit testing.
         */
        void set_testing_without_socket(bool value)
        {
            testing_without_socket_ = value;
        }


    };
}

#endif  // ZEROMQ_WRAPPER
