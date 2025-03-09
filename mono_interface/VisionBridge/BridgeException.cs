/* ****************************************************************
 * Vision Bridge
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: Aug. 2015
 * 
 * Developed under contract for:
 * Academy Center for UAS Research 
 * Department of Electrical and Computer Engineering 
 * HQ USAFA/DFEC 
 * 2354 Fairchild Drive 
 * USAF Academy, CO 80840-6236 
 *******************************************************************/
using System;

namespace VisionBridge
{
    public class BridgeException : Exception
    {
        public BridgeException()
        {
        }

        public BridgeException(string message)
            :base(message)
        {
        }

        public BridgeException(string message, Exception inner)
            :base(message, inner)
        {
        }
    }
}

