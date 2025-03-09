//
// Created by wavelet on 7/22/16.
//

#ifndef VIDERE_DEV_LOGGER_H
#define VIDERE_DEV_LOGGER_H

/* Macros are defined hear to make things cleaner.
 * detangle the dependency of the code on log4cxx a bit.
 * These assume "log4cpp_" within the scope of macro.
 */
#include <log4cxx/logger.h>

#define LOGTRACE( message )\
        LOG4CXX_TRACE(log4cpp_, message );
#define LOGDEBUG( message )\
        LOG4CXX_DEBUG(log4cpp_, message );
#define LOGINFO( message )\
        LOG4CXX_INFO(log4cpp_, message );
#define LOGWARN( message )\
        LOG4CXX_WARN(log4cpp_, message );
#define LOGERROR( message )\
        LOG4CXX_ERROR(log4cpp_, message );
#define LOGFATAL( message )\
        LOG4CXX_FATAL(log4cpp_, message );


#endif //VIDERE_DEV_LOGGER_H
