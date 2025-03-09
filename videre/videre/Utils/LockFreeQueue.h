/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Jan. 13, 2018
 *
 * NeuroGroove IMU Communications Interface
 *******************************************************************/


#ifndef VIDERE_DEV_LOCKFREEQUEUE_H
#define VIDERE_DEV_LOCKFREEQUEUE_H

#include <string>
#include <memory.h>

namespace dtiUtils
{
    //This lock-free queue can be safely used between two
    //thread with one producer and one consumer.  The use of
    //the user must follow the specific requrements to ensure
    //the thread-safety.
    //Items are added to the Head by the producer, and removed
    //from the tail by the consumer.
    //The producer must fillin the data/object at the head before
    //incrementing the head.  The Producter must verify the
    //queue is not full before adding new data or it will be lost.
    //The consumer must extract the data/object at the tail before
    //incrementing the tail.  The consumer must verify that the
    //queue is not empty before extracting data/object at the tail.
    template <class T>
    class LockFreeQueue
    {
#define MAXQUEUESIZE 8192;

        //The Queue array of the given type.
    private:

        T *_queue = nullptr;

        int _qsize;
        volatile int _head = 0;
        volatile int _tail = 0;

    public:
        LockFreeQueue() {}

        LockFreeQueue(int queueSize)
        {
            init(queueSize);
        }

        ~LockFreeQueue()
        {
            if(_queue != nullptr)
            {
                delete[]_queue;
                _queue = nullptr;
            }
        }

        void init(int queueSize)
        {
            if(_queue == nullptr)
            {
                int max = MAXQUEUESIZE;
                _qsize = queueSize < 2 ? 2 : queueSize > max ? max : queueSize;
                _queue = new T[_qsize];
            }
        }

        //The user is responsible for creating the elements of the Queue.
        //There is probably a beter way to do this... but I don't know what
        //it is at this time.
        //The user is also responsible for cleaning up the memory
        //allocated to queue elements.
        T* getQueuePtr()
        {
            return _queue;
        }

        int getQueueSize() {return _qsize;}

        //May be called by either thread.
        //The Queue is empty if the tail equals the head.
        bool IsQueueEmpty()
        {
            return _tail == _head;
        }

        bool IsQueueFull()
        {
            int tmpHead = (_head + 1) % _qsize;
            return tmpHead == _tail;
        }

        //Only the Consumer may safely use this
        //call... it could be used if the queue is full
        //and the consumer wants to clear all old messages.
        void ClearQueue()
        {
            _tail = _head;
        }

        T& getHeadItemReference()
        {
            return _queue[_head];
        }

        T& getTailItemReference()
        {
            return _queue[_tail];
        }

        //May only be called by the producer.
        //returns true if the queue is full...
        //which means the head was not incremented.
        bool IncrementHead()
        {
            bool full = true;
            int tmpHead = (_head + 1) % _qsize;
            if(tmpHead != _tail)
            {
                _head = tmpHead;
                full = false;
            }
            return full;
        }

        //May only be called by the consumer.
        //returns true if the queue is empty after the increment.
        //retuns false if the queue is not empty.
        bool IncrementTail()
        {
            bool empty = true;
            if(_tail != _head)
            {
                int tmpTail = (_tail + 1) % _qsize;
                _tail = tmpTail;
                empty = _tail == _head;
            }
            return empty;
        }

    };




}





#endif //VIDERE_DEV_LOCKFREEQUEUE_H
