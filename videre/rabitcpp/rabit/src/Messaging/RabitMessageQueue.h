/* ****************************************************************
 * Rabit Multi-Threaded Management System
 * Athrs: Randal Direen PhD
 *        Harry Direen PhD
 * www.direentech.com
 * Date: June 2016
 *
 *******************************************************************/

#ifndef RABIT_MESSAGE_QUEUE
#define RABIT_MESSAGE_QUEUE

#include <string>
#include <memory>
#include <boost/signals2.hpp>
#include "SafeQueue.h"

namespace Rabit
{

    //The Rabit Message Queue is a wrapper for Std Lib Queue.
    //The Rabit Message Queue adds Mutex locking for thread safety.
    //The Rabit Message Queue adds event signals which can be used to indicate
    //that a message has been added or removed from the queue.
    template<class T>
    class RabitMessageQueue
    {

    private:
        const int UPPER_BOUND = 1000000;        //Absolute max size of the queue.

        std::string _msgQueueName;

        std::unique_ptr<SafeQueue<T>> _messQueue;

        boost::signals2::signal<void()> _sigEnqueue;
        boost::signals2::signal<void()> _sigDequeue;

    public:
        typedef boost::signals2::signal<void()>::slot_type _sigEnqueuSlotType;

        // Accessors
    public:
        std::string GetMessageQueueName()
        {
            return _msgQueueName;
        }

        int GetMaxNoMessagesAllowedInQueue()
        {
            return _messQueue->GetMaxQueueSize();
        }


        // Methods
    public:
        //Instantiate a new Queue... the maxNoMessages is the maximum
        //number of items the Queue can hold.
        //The msgQName is the the name the queue will be accessed by.
        RabitMessageQueue(int maxNoMessages, std::string msgQName)
        {
             _msgQueueName = msgQName.empty() ? "MessageQueue" : msgQName;

            maxNoMessages = maxNoMessages < 1 ? 1 : maxNoMessages;
            maxNoMessages = maxNoMessages > UPPER_BOUND ? UPPER_BOUND : maxNoMessages;

            _messQueue = std::unique_ptr<SafeQueue<T>>(new SafeQueue<T>(maxNoMessages));
        }


        //Register a Event Handler for when a message is added to the queue.
        void Register_SomethingEnqueued(const boost::function<void()> &handler)
        {
            _sigEnqueue.connect(handler);
        }

        //Register an event handler for when an item is pulled from the queue.
        void Register_SomethingDequeued(const boost::function<void()> &handler)
        {
            _sigDequeue.connect(handler);
        }


        void ClearMessageQueue()
        {
            _messQueue->clear();
        }

        int NoMessagesInQueue()
        {
            return _messQueue->size();
        }


        //Check to see if the Queue is currently empty
        //Returns true if empty, false otherwise.
        bool IsEmpty()
        {
            return _messQueue->empty();
        }

        //Add a message to the Queue.
        //Returns true if the message was added to the queue...
        //returns false if the the queue is full and the message could not be
        //added.
        bool AddMessage(T msg)
        {
            bool msgAddedToQueue = _messQueue->enqueue(msg);
            _sigEnqueue();
            return msgAddedToQueue;
         }


        //Add a message to the Queue.
        //Returns true if the message was added to the queue...
        //returns false if the the queue is full and the message could not be
        //added.
        //This method does not trigger a push-message event.
        bool AddMessageNoEventTrigger(T msg)
        {
            return _messQueue->enqueue(msg);
        }

        //Get a message from the Message Queue.
        //Returns true if a message is obtained.
        //Returns false if the message queue is empty.
        bool GetMessage(T &msg)
        {
            bool msgObtained = _messQueue->dequeue(msg);
            if(msgObtained)
            {
                _sigDequeue();
            }
            return msgObtained;
        }

    };

}


#endif //RABIT_MESSAGE_QUEUE
