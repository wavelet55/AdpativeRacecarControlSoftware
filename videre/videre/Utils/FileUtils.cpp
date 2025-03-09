/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Oct. 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#include "FileUtils.h"
#include <iomanip>
#include <boost/algorithm/string/compare.hpp>
#include "boost/date_time.hpp"
#include <iostream>
#include <string>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace boost::gregorian;
using namespace boost::posix_time;

namespace VidereFileUtils
{

    int CompareCharStringsCaseInsensitive(char const *lhs, char const *rhs)
    {
        for (; *lhs != '\0' && *rhs != '\0'; ++lhs, ++rhs)
        {
            if (tolower(*lhs) != tolower(*rhs))
            {
                return (tolower(*lhs) < tolower(*rhs) ? -1 : 1);
            }
        }
        if (*lhs == '\0' && *rhs == '\0')
            return 0;
        else if (*lhs == '\0')
            return -1;
        else
            return 1;
    }

    int CompareStringCaseInsensitive(const string &lhStr, const string &rhStr)
    {
        char const *lhs = lhStr.c_str();
        char const *rhs = rhStr.c_str();
        return CompareCharStringsCaseInsensitive(lhs, rhs);
    }

    //Return true if lhStr < rhStr, otherwise false.
    bool SortCompareStringCaseInsensitive(const std::string &lhStr, const std::string &rhStr)
    {
        int cmpVal = CompareStringCaseInsensitive(lhStr, rhStr);
        return cmpVal < 0 ? true : false;
    }

    int ComparePathCaseInsensitive(const path &lhStr, const path &rhStr)
    {
        char const *lhs = lhStr.c_str();
        char const *rhs = rhStr.c_str();
        return CompareCharStringsCaseInsensitive(lhs, rhs);
    }

    bool SortComparePathCaseInsensitive(const path &lhStr, const path &rhStr)
    {
        int cmpVal = ComparePathCaseInsensitive(lhStr, rhStr);
        return cmpVal < 0 ? true : false;
    }


    int GetListFilesInDirectory(vector<path> *resultListOfFilesPtr,
                                const string &directory,
                                const string &fileExt,
                                const std::string &baseFilename,
                                bool sortFilenames)
    {

        resultListOfFilesPtr->clear();
        path fileExtWDot = "";
        bool checkFileExt = false;
        bool checkBaseFilename = false;

        try
        {
            if (fileExt.length() > 0)
            {
                checkFileExt = true;
                if (fileExt[0] != '.')
                    fileExtWDot = "." + fileExt;
                else
                    fileExtWDot = fileExtWDot;

                if (baseFilename.length() > 0)
                {
                    checkBaseFilename = true;
                }
            }


            directory_iterator it(directory);
            directory_iterator endit;
            while (it != endit)
            {
                if (is_regular_file(*it))
                {
                    if (checkFileExt)
                    {
                        if (ComparePathCaseInsensitive(it->path().extension(), fileExtWDot) == 0)
                        {
                            if(checkBaseFilename)
                            {
                                if(it->path().filename().string().find(baseFilename) == 0)
                                {
                                    resultListOfFilesPtr->push_back(it->path());
                                }
                            }
                            else
                            {
                                resultListOfFilesPtr->push_back(it->path());
                            }
                        }
                    }
                    else
                    {
                        resultListOfFilesPtr->push_back(it->path());
                    }
                }
                ++it;
            }
            if (sortFilenames)
            {
                sort(resultListOfFilesPtr->begin(), resultListOfFilesPtr->end(), SortComparePathCaseInsensitive);
            }
        }
        catch (std::exception &e)
        {
            //LOGERROR("StreamRecordManager: Exception: " << e.what());
        }
        return resultListOfFilesPtr->size();
    }

    int GetListFilesInDirectoryRecursive(vector<path> *resultListOfFilesPtr,
                                                  const string &directory,
                                                  const string &fileExt,
                                                  bool sortFilenames)
    {

        resultListOfFilesPtr->clear();
        path fileExtWDot = "";
        bool checkFileExt = false;

        try
        {
            if (fileExt.length() > 0)
            {
                checkFileExt = true;
                if (fileExt[0] != '.')
                    fileExtWDot = "." + fileExt;
                else
                    fileExtWDot = fileExtWDot;
            }

            recursive_directory_iterator it(directory);
            recursive_directory_iterator endit;
            while (it != endit)
            {
                if (is_regular_file(*it))
                {
                    if (checkFileExt)
                    {
                        if (ComparePathCaseInsensitive(it->path().extension(), fileExtWDot) == 0)
                            resultListOfFilesPtr->push_back(it->path());
                    } else
                    {
                        resultListOfFilesPtr->push_back(it->path());
                    }
                }
                ++it;
            }
            if (sortFilenames)
            {
                sort(resultListOfFilesPtr->begin(), resultListOfFilesPtr->end(), SortComparePathCaseInsensitive);
            }
        }
        catch (std::exception &e)
        {
            //LOGERROR("StreamRecordManager: Exception: " << e.what());
        }
        return resultListOfFilesPtr->size();
    }


    bool CheckDirectoryExits(const string &directory)
    {
        bool dirExits = false;
        path dirpath(directory);
        return boost::filesystem::is_directory(dirpath);
    }

    bool CheckDirectoryExits(const path &dirpath)
    {
        bool dirExits = false;
        return boost::filesystem::is_directory(dirpath);
    }


    bool CreateDirectory(const path &dirpath)
    {
        bool dirExits = false;
        if (!is_directory(dirpath))
        {
            if (boost::filesystem::create_directories(dirpath))
            {
                dirExits = true;
            }
        } else
        {
            dirExits = true;
        }
        return dirExits;
    }

    bool CreateDirectory(const string &directory)
    {
        bool dirExits = false;
        path dirpath(directory);
        return CreateDirectory(dirpath);
    }

    bool DeleteDirectoryAndAllFiles(const std::string &directory)
    {
        bool filesDeleted = true;
        if(CheckDirectoryExits(directory))
        {
            try
            {
                boost::filesystem::remove_all(directory);
                filesDeleted = true;
            }
            catch(std::exception &e)
            {
                filesDeleted = false;
            }
        }
        return filesDeleted;
    }


    std::string DateTimeToISOString(const ptime &dt, bool dateOnly)
    {
        char dtbuf[64];
        int idx = sprintf(dtbuf, "%.4i%.2i%.2i",(int)dt.date().year(),
                          (int)dt.date().month().as_number(),
                          (int)dt.date().day());
        if(!dateOnly)
        {
            idx += sprintf(dtbuf + idx, "T%.2i%.2i%.2i",
                           (int)dt.time_of_day().hours(),
                           (int)dt.time_of_day().minutes(),
                           (int)dt.time_of_day().seconds());
        }
        string dtStr(dtbuf,idx);
        return dtStr;
    }

    //Add the current Time/Data stamp in the form: _2016-10-23T15:23:46
    //If dateOnly is true, the time is not added.
    std::string AddCurrentTimeDateStampToString(const std::string &strVal, bool dateOnly)
    {
        std::string nameDTString = strVal + "_";
        //date_time dtStamp;
        ptime dtstamp = microsec_clock::local_time();
        string dtStr = DateTimeToISOString(dtstamp, dateOnly);
        nameDTString += dtStr;
        return nameDTString;
    }

    std::string AddIndexToFilename(string filename, int idx,
                                   int fixedIdxNoDigits, string ext)
    {
        string fullfn = filename + "_";
        char fnBuf[256];
        if(fullfn.length() + ext.length() + 12 < 256)
        {
            int strSize = 0;
            fixedIdxNoDigits = fixedIdxNoDigits < 1 ? 1 : fixedIdxNoDigits > 6 ? 6 : fixedIdxNoDigits;
            switch (fixedIdxNoDigits)
            {
                case 1:
                    strSize = sprintf(fnBuf, "%s%d", fullfn.c_str(), idx);
                    break;
                case 2:
                    strSize = sprintf(fnBuf, "%s%.2i", fullfn.c_str(), idx);
                    break;
                case 3:
                    strSize = sprintf(fnBuf, "%s%.3i", fullfn.c_str(), idx);
                    break;
                case 4:
                    strSize = sprintf(fnBuf, "%s%.4i", fullfn.c_str(), idx);
                    break;
                case 5:
                    strSize = sprintf(fnBuf, "%s%.5i", fullfn.c_str(), idx);
                    break;
                case 6:
                    strSize = sprintf(fnBuf, "%s%.6i", fullfn.c_str(), idx);
                    break;
                default:
                    strSize = sprintf(fnBuf, "%s%d", fullfn.c_str(), idx);
                    break;
            }
            fullfn = string(fnBuf, strSize);
        }
        else
        {
            fullfn += to_string(idx);
        }
        if (ext.length() > 0)
        {
            if(ext[0] != '.')
                fullfn += "." + ext;
            else
                fullfn += ext;
        }
       return fullfn;
    }

    //Add file extention if it does not already exist.
    std::string AddOrReplaceFileExtention(const std::string filename, const std::string ext)
    {
        std::string fext = ext;
        std::string outpFn = filename;
        if( ext.size() > 0)
        {
            if (ext[0] != '.')
                fext = "." + ext;

            int dotPos = filename.find_last_of('.');
            if( dotPos > 0 )
            {
                outpFn = filename.substr(0, dotPos);
            }
            outpFn = outpFn + fext;
        }
        return outpFn;
    }

    std::string GetImageFileExtention(videre::ImageFormatType_e imgFmtType)
    {
        std::string ext = "jpg";
        switch(imgFmtType)
        {
            case videre::ImageFormatType_e::ImgFType_JPEG:
                ext = "jpg";
                break;
            case videre::ImageFormatType_e::ImgFType_Raw:
                ext = "raw";
                break;
        }
        return ext;
    }

    std::string byteArrayToString(unsigned char* data, int size)
    {
        char txt[1024];
        string str;
        int N = size > 1023 ? 1023 : size;
        int n = 0;
        for(n = 0; n < N; n++)
        {
            char c = (char)data[n];
            if(c != 0)
            {
                txt[n] = (char) data[n];
            }
            else
            {
                break;
            }
        }
        txt[n] = 0;
        str = txt;
        return str;
    }

}
