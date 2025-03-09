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

#include "zeromq_wrapper.h"
#include "zeromq_wrapper_exception.h"

using namespace std;
using namespace zmq;

namespace videre
{

    ZeroMQWrapper::ZeroMQWrapper()
    {

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        zmq_initialized_ = false;
        testing_without_socket_ = false;

        //reply_zmqmsg_sptr_ = unique_ptr<zmq::message_t>(new zmq::message_t());
        //publish_visionresults_zmqmsg_sptr_ = unique_ptr<zmq::message_t>(new zmq::message_t());
        //publish_image_zmgmsg_sptr_ = unique_ptr<zmq::message_t>(new zmq::message_t());
        //subscribe_telemetry_zmqmsg_sptr_ = unique_ptr<zmq::message_t>(new zmq::message_t());
        //publish_monitor_zmqmsg_sptr_ = unique_ptr<zmq::message_t>(new zmq::message_t());
    }

    void ZeroMQWrapper::Initialize(string host_rep, string host_pub,
                                   string host_pub_vid, string host_sub_tel,
                                   string host_pub_mon,
                                   string host_pub_nexus_bci_port,
                                   string host_sub_nexus_bci_port)
    {
        int hwmVal = 0;
        try
        {
            zcontext_sptr_ = unique_ptr<zmq::context_t>(new zmq::context_t(1));
            zsocket_cmdresponse_sptr_ = unique_ptr<zmq::socket_t>(new zmq::socket_t(*zcontext_sptr_, ZMQ_REP));
            zsocket_pub_visionresults_sptr_ = unique_ptr<zmq::socket_t>(new zmq::socket_t(*zcontext_sptr_, ZMQ_PUB));
            zsocket_pub_videoout_sptr_ = unique_ptr<zmq::socket_t>(new zmq::socket_t(*zcontext_sptr_, ZMQ_PUB));
            zsocket_pub_visionsysmonitor_sptr_ = unique_ptr<zmq::socket_t>(new zmq::socket_t(*zcontext_sptr_, ZMQ_PUB));
            zsocket_sub_telemeteryinput_sptr_ = unique_ptr<zmq::socket_t>(new zmq::socket_t(*zcontext_sptr_, ZMQ_SUB));
            zsocket_sub_telemeteryinput_sptr_->setsockopt(ZMQ_SUBSCRIBE, "", 0);
            zsocket_pub_nexus_bci_sptr_ = unique_ptr<zmq::socket_t>(new zmq::socket_t(*zcontext_sptr_, ZMQ_PUB));
            zsocket_sub_nexus_bci_sptr_ = unique_ptr<zmq::socket_t>(new zmq::socket_t(*zcontext_sptr_, ZMQ_SUB));

            /* Bind to local host*/
            zsocket_cmdresponse_sptr_->bind(host_rep.c_str());

            zsocket_pub_visionresults_sptr_->bind(host_pub.c_str());

            zsocket_pub_videoout_sptr_->bind(host_pub_vid.c_str());
            //Set a low High-water mark so that this socket does not build up
            //video messages if the receiver is slow pulling them.
            //hwmVal = 5;
            //zsocket_pub_videoout_sptr_->setsockopt(ZMQ_HWM, &hwmVal, sizeof(int));

            zsocket_pub_visionsysmonitor_sptr_->bind(host_pub_mon.c_str());
            //Set a  High-water mark so that this socket does not build up
            //video messages if the receiver is slow pulling them.
            //hwmVal = 25;
            //zsocket_pub_visionsysmonitor_sptr_->setsockopt(ZMQ_HWM, &hwmVal, sizeof(int));

            zsocket_sub_telemeteryinput_sptr_->connect(host_sub_tel.c_str());

            zsocket_pub_nexus_bci_sptr_->bind(host_pub_nexus_bci_port.c_str());

            //Subscribe to all topics
            std::string alltopics = "";
            zsocket_sub_nexus_bci_sptr_->bind(host_sub_nexus_bci_port.c_str());
            zsocket_sub_nexus_bci_sptr_->setsockopt(ZMQ_SUBSCRIBE, alltopics.c_str(), alltopics.length());;

        } 
        catch (exception &e)
        {
            // Recast the exception to be one of ours.
            LOGERROR("ZeroMQWrapper::Initialize exception: " << e.what())
            throw ZeroMQWrapperException(e.what());
        }
        zmq_initialized_ = true;
    }

    bool ZeroMQWrapper::ReceiveRequestFromHOPS(zmq::message_t *req_ptr)
    {
        return zsocket_cmdresponse_sptr_->recv(req_ptr, ZMQ_NOBLOCK);
    }

    bool ZeroMQWrapper::SendResponse(std::string response)
    {

        if (zmq_initialized_)
        {
            int slength = response.length();

            //TODO: I'm still not sure what exactly this does? I want to make
            //sure I'm not allocating new memory all the time.
            _reply_zmqmsg.rebuild(slength);
            memcpy((void *) _reply_zmqmsg.data(), response.c_str(), slength);

            /* returns true if message sent successfully */
            return SendOnSocketRep();
        } else
        {
            LOGERROR("ZeroMQWrapper::SendResponse: The 0MQ sockets are not initialized for sending messages" );
            //throw ZeroMQWrapperException("The 0MQ sockets are not initialized for sending messages.");
            return true;
        }
    }

    bool ZeroMQWrapper::PublishVisionProcResults(std::string publish)
    {

        if (zmq_initialized_)
        {
            int slength = publish.length();

            //TODO: I'm still not sure what exactly this does? I want to make
            //sure I'm not allocating new memory all the time.
            _publish_visionresults_zmqmsg.rebuild(slength);
            memcpy((void *) _publish_visionresults_zmqmsg.data(), publish.c_str(), slength);

            /* returns true if message sent successfully */
            return SendOnSocketPubVisionProcResults();
        } else
        {
            LOGERROR("ZeroMQWrapper::Publish: The 0MQ sockets are not initialized for sending messages" );
            throw ZeroMQWrapperException("The 0MQ sockets are not initialized for sending messages.");
        }
    }

    bool ZeroMQWrapper::PublishImageMessage(string imgMsg)
    {

        if (zmq_initialized_)
        {
            int slength = imgMsg.size();

            //TODO: I'm still not sure what exactly this does? I want to make
            //sure I'm not allocating new memory all the time.
            _publish_image_zmgmsg.rebuild(slength);
            memcpy((void *) _publish_image_zmgmsg.data(), imgMsg.c_str(), slength);

            /* returns true if message sent successfully */
            return SendOnSocketPubVideo();
        } else
        {
            LOGERROR("ZeroMQWrapper::PublishVideo: The 0MQ sockets are not initialized for sending messages" );
            throw ZeroMQWrapperException("The 0MQ sockets are not initialized for sending messages.");
        }
    }

    bool ZeroMQWrapper::PublishMonitor(string json_data)
    {

        if (zmq_initialized_)
        {
            int slength = json_data.size();

            _publish_monitor_zmqmsg.rebuild(slength);
            //publish_monitor_zmqmsg_sptr_->
            memcpy((void *) _publish_monitor_zmqmsg.data(), json_data.c_str(), slength);

            /* returns true if message sent successfully */
            return SendOnSocketPubMonitor();
        } else
        {
            LOGERROR("ZeroMQWrapper::PublishMonitor: The 0MQ sockets are not initialized for sending messages" );
            throw ZeroMQWrapperException("The 0MQ sockets are not initialized for sending messages.");
        }
    }

    bool ZeroMQWrapper::SubscribedTelemetryFromHOPS(zmq::message_t *sub_ptr)
    {

        bool got_message = false;
        int events = 0;
        size_t events_size = sizeof(int);

        // Priming read
        got_message = zsocket_sub_telemeteryinput_sptr_->recv(sub_ptr, ZMQ_NOBLOCK);

        // See if there are more messages to read
        zsocket_sub_telemeteryinput_sptr_->getsockopt(ZMQ_EVENTS, static_cast<void *>(&events), &events_size);
        while (events & ZMQ_POLLIN)
        {

            // Receive the new (and perhaps most recent) message
            got_message = zsocket_sub_telemeteryinput_sptr_->recv(sub_ptr);

            // Poll again for additional messages
            zsocket_sub_telemeteryinput_sptr_->getsockopt(ZMQ_EVENTS, static_cast<void *>(&events), &events_size);
        }

        return got_message;
    }

    bool ZeroMQWrapper::PublishNexusBCI(uint8_t *dataBuf, int dataLength)
    {

        if (zmq_initialized_)
        {
            _publish_nexus_bci_zmqmsg.rebuild(dataLength);
            //publish_monitor_zmqmsg_sptr_->
            memcpy((void *) _publish_nexus_bci_zmqmsg.data(), dataBuf, dataLength);

            /* returns true if message sent successfully */
            return SendOnSocketPubNexusBCI();
        }
        else
        {
            LOGERROR("ZeroMQWrapper::PublishMonitor: The 0MQ sockets are not initialized for sending messages" );
            throw ZeroMQWrapperException("The 0MQ sockets are not initialized for sending messages.");
        }
    }

    bool ZeroMQWrapper::ReceiveNexusBCIMessage(zmq::message_t *sub_ptr)
    {
        bool got_message = false;
        int events = 0;
        size_t events_size = sizeof(int);

        // Priming read
        got_message = zsocket_sub_nexus_bci_sptr_->recv(sub_ptr, ZMQ_NOBLOCK);

        // See if there are more messages to read
        //zsocket_sub_nexus_bci_sptr_->getsockopt(ZMQ_EVENTS, static_cast<void *>(&events), &events_size);
        //while (events & ZMQ_POLLIN)
        //{

        //    // Receive the new (and perhaps most recent) message
        //    got_message = zsocket_sub_nexus_bci_sptr_->recv(sub_ptr);

        //    // Poll again for additional messages
        //    zsocket_sub_nexus_bci_sptr_->getsockopt(ZMQ_EVENTS, static_cast<void *>(&events), &events_size);
        //}

        return got_message;
    }

    void ZeroMQWrapper::Close()
    {
        try
        {
            zsocket_cmdresponse_sptr_->close();
            zsocket_pub_visionresults_sptr_->close();
            zsocket_pub_videoout_sptr_->close();
            zsocket_sub_telemeteryinput_sptr_->close();
        } catch (exception &e)
        {
            // Recast the exception to be one of ours.
            throw ZeroMQWrapperException(e.what());
        }
    }

    /*****************************************************************************
     *
     * Implementation of private members:
     *
     *****************************************************************************/

    bool ZeroMQWrapper::SendOnSocketRep()
    {

        if (!testing_without_socket_)
        {
            /* returns true if message sent successfully */
            int bytesSent = zsocket_cmdresponse_sptr_->send(_reply_zmqmsg);
            if(bytesSent <= 0)
            {
                LOGERROR("ZMQ Response Socket Send Error No: " << bytesSent);
            }
            return bytesSent > 0;
        }
        else
        {
            return false;
        }
    }

    bool ZeroMQWrapper::SendOnSocketPubVisionProcResults()
    {
        if (!testing_without_socket_)
        {
            /* returns true if message sent successfully */
            int bytesSent = zsocket_pub_visionresults_sptr_->send(_publish_visionresults_zmqmsg);
            if(bytesSent <= 0)
            {
                LOGERROR("ZMQ PubVisionProc Socket Send Error No: " << bytesSent);
            }
            return bytesSent > 0;
        }
        else
        {
            return false;
        }
    }

    bool ZeroMQWrapper::SendOnSocketPubVideo()
    {
        if (!testing_without_socket_)
        {
            /* returns true if message sent successfully */
            int bytesSent =  zsocket_pub_videoout_sptr_->send(_publish_image_zmgmsg);
            if(bytesSent <= 0)
            {
                LOGERROR("ZMQ PubVideo Socket Send Error No: " << bytesSent);
            }
            return bytesSent > 0;
        } 
        else
        {
            return false;
        }
    }

    bool ZeroMQWrapper::SendOnSocketPubMonitor()
    {
        if (!testing_without_socket_)
        {
            /* returns true if message sent successfully */
            int bytesSent =  zsocket_pub_visionsysmonitor_sptr_->send(_publish_monitor_zmqmsg);
            if(bytesSent <= 0)
            {
                LOGERROR("ZMQ PubMonitor Socket Send Error No: " << bytesSent);
            }
            return bytesSent > 0;
        } 
        else
        {
            return false;
        }
    }

    bool ZeroMQWrapper::SendOnSocketPubNexusBCI()
    {
        if (!testing_without_socket_)
        {
            /* returns true if message sent successfully */
            int bytesSent =  zsocket_pub_nexus_bci_sptr_->send(_publish_nexus_bci_zmqmsg);
            if(bytesSent <= 0)
            {
                LOGERROR("ZMQ Nexus BCI Socket Send Error No: " << bytesSent);
            }
            return bytesSent > 0;
        } 
        else
        {
            return false;
        }
    }

}
