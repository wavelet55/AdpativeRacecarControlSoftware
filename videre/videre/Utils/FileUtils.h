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


#ifndef VIDERE_DEV_FILEUTILS_H
#define VIDERE_DEV_FILEUTILS_H

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <boost/filesystem.hpp>
#include "boost/date_time.hpp"
#include "global_defines.h"


namespace VidereFileUtils
{

    //Return true if -1 if lhStr < rhStr, +1 if lhStr > rhStr, 0 if equal.
    int CompareCharStringsCaseInsensitive(char const *lhs, char const *rhs);

    //Return true if -1 if lhStr < rhStr, +1 if lhStr > rhStr, 0 if equal.
    int CompareStringCaseInsensitive(const std::string &lhStr, const std::string & rhStr);

    //Return true if lhStr < rhStr, otherwise false.
    bool SortCompareStringCaseInsensitive(const std::string &lhStr, const std::string & rhStr);

    //Return true if lhStr < rhStr, otherwise false.
    int ComparePathCaseInsensitive(const boost::filesystem::path &lhStr, const boost::filesystem::path & rhStr);

    bool SortComparePathCaseInsensitive(const boost::filesystem::path &lhStr, const boost::filesystem::path & rhStr);

    //Get a list of the full file names in a directory.  Set fileExt to the a file extention
    //so that only files with that extension are retrieved... or set it to: "".
    //May also filter on a base filename (requires file extention to be set or set it to: "".
    // If sort = true a case-insensitive file sort will be done,
    int GetListFilesInDirectory(std::vector<boost::filesystem::path> *resultListOfFilesPtr,
                                const std::string &directory,
                                 const std::string &fileExt = "",
                                 const std::string &baseFilename = "",
                                 bool sortFilenames = false);

    //Get a list of the full file names in a directory, recurse into subdirectories.
    //Set fileExt to the a file extention
    //so that only files with that extension are retrieved... or set it to: "".
    // If sort = true a case-insensitive file sort will be done,
    int GetListFilesInDirectoryRecursive(std::vector<boost::filesystem::path> *resultListOfFilesPtr,
                                        const std::string &directory,
                                          const std::string &fileExt = "",
                                          bool sortFilenames = false);

    std::string DateTimeToISOString(const boost::posix_time::ptime &dt, bool dateOnly = false);

    //Add the current Time/Data stamp in the form: _2016-10-23T15:23:46
    //If dateOnly is true, the time is not added.
    std::string AddCurrentTimeDateStampToString(const std::string &strVal, bool dateOnly = false);

    std::string AddIndexToFilename(std::string filename, int idx,
                                   int fixedIdxNoDigits,std::string ext);

    //Add or replace file extention if it does not already exist.
    std::string AddOrReplaceFileExtention(const std::string filename, const std::string ext);

    //Check to see if the directory exits... returns true if it exists, false otherwise;
    bool CheckDirectoryExits(const std::string &directory);

    //Check to see if the directory exits... returns true if it exists, false otherwise;
    bool CheckDirectoryExits(const boost::filesystem::path &directory);

    //Create the directory if it does not already exist.
    //Returns true if created or if it exits, false otherwise.
    bool CreateDirectory(const std::string &directory);

    //Create the directory if it does not already exist.
    //Returns true if created or if it exits, false otherwise.
    bool CreateDirectory(const boost::filesystem::path &directory);

    bool DeleteDirectoryAndAllFiles(const std::string &directory);

    std::string GetImageFileExtention(videre::ImageFormatType_e imgFmtType);

    std::string byteArrayToString(unsigned char* data, int size);

    class FileUtils
    {

    };

}

#endif //VIDERE_DEV_FILEUTILS_H
