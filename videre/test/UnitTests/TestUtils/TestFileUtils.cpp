//
// Created by wavelet on 10/4/16.
//

#include "TestFileUtils.h"

#include <memory>
#include <string>
#include <boost/filesystem.hpp>
#include "boost/date_time.hpp"

using namespace std;
using namespace VidereFileUtils;
using namespace boost::filesystem;
using namespace boost::gregorian;
using namespace boost::posix_time;

TEST_F(TestFileUtils, TestStringCaseInsensitiveCompare)
{
    string str1 = "DogFight";
    string str2 = "DogFighting";
    string str3 = "DogCatFight";
    string str4 = "dogfight";

    int cmp1 = CompareStringCaseInsensitive(str1, str2);
    //EXPECT_EQ(cmp1, -1);

    int cmp2 = CompareStringCaseInsensitive(str2, str1);
    //EXPECT_EQ(cmp2, 1);

    int cmp3 = CompareStringCaseInsensitive(str1, str3);
    //EXPECT_EQ(cmp3, -1);

    int cmp4 = CompareStringCaseInsensitive(str1, str4);
    //EXPECT_EQ(cmp3, 0);

}

TEST_F(TestFileUtils, TestPathCaseInsensitiveCompare)
{
    boost::filesystem::path str1 = "DogFight";
    boost::filesystem::path str2 = "DogFighting";
    boost::filesystem::path str3 = "DogCatFight";
    boost::filesystem::path str4 = "dogfight";

    int cmp1 = ComparePathCaseInsensitive(str1, str2);
    //EXPECT_EQ(cmp1, -1);

    int cmp2 = ComparePathCaseInsensitive(str2, str1);
    //EXPECT_EQ(cmp2, 1);

    int cmp3 = ComparePathCaseInsensitive(str1, str3);
    //EXPECT_EQ(cmp3, -1);

    int cmp4 = ComparePathCaseInsensitive(str1, str4);
    //EXPECT_EQ(cmp3, 0);

}

TEST_F(TestFileUtils, TestPathCaseInsensitiveSort)
{
    vector<boost::filesystem::path> filenames = vector<boost::filesystem::path>();
    boost::filesystem::path str1 = "Video_124.vid";
    boost::filesystem::path str2 = "Video_345.vid";
    boost::filesystem::path str3 = "video_125.vid";
    boost::filesystem::path str4 = "viDeo_530.vid";

    filenames.push_back(str2);
    filenames.push_back(str4);
    filenames.push_back(str1);
    filenames.push_back(str3);

    sort(filenames.begin(), filenames.end(), SortComparePathCaseInsensitive);

    int cmpVal = ComparePathCaseInsensitive(filenames[0], str1);
    //EXPECT_EQ(cmpVal, 0);
    cmpVal = ComparePathCaseInsensitive(filenames[1], str3);
    //EXPECT_EQ(cmpVal, 0);
    cmpVal = ComparePathCaseInsensitive(filenames[2], str2);
    //EXPECT_EQ(cmpVal, 0);
    cmpVal = ComparePathCaseInsensitive(filenames[3], str4);
    //EXPECT_EQ(cmpVal, 0);

}

TEST_F(TestFileUtils, TestReadDirectory)
{
    string dirname = "testfiles";
    string fileExt = "test";
    vector<boost::filesystem::path> filenames;
    int numOfFiles = 0;

    numOfFiles = GetListFilesInDirectory(&filenames, dirname, fileExt, "basename", true);

    if(numOfFiles >= 3)
    {
        int cmpVal = ComparePathCaseInsensitive(filenames[0], "testfiles/config.ini.test");
        //EXPECT_EQ(cmpVal, 0);
        cmpVal = ComparePathCaseInsensitive(filenames[1], "testfiles/config_bad1.ini.test");
        //EXPECT_EQ(cmpVal, 0);
        cmpVal = ComparePathCaseInsensitive(filenames[2], "testfiles/config_bad2.ini.test");
        //EXPECT_EQ(cmpVal, 0);
    }
}

TEST_F(TestFileUtils, TestCreateDirectory)
{
    string dir1name = "testdir/dir1";

    bool dirExist = CheckDirectoryExits(dir1name);
    if(dirExist)
    {
        boost::filesystem::remove(dir1name);
    }

    dirExist = CreateDirectory(dir1name);
    //EXPECT_EQ(dirExist, true);
    dirExist = CheckDirectoryExits(dir1name);
    //EXPECT_EQ(dirExist, true);
    if(dirExist)
    {
        boost::filesystem::remove(dir1name);
    }
}

TEST_F(TestFileUtils, TestDateTimeToISOString)
{
    ptime dtstamp = microsec_clock::local_time();
    string dtStr = DateTimeToISOString(dtstamp, false);
    cout << dtStr << endl;

    dtstamp = microsec_clock::local_time();
    dtStr = DateTimeToISOString(dtstamp, true);
    cout << dtStr << endl;
}

TEST_F(TestFileUtils, TestAddCurrentTimeDateStampToString)
{
    string dirname("TstDirname");
    string dirnamePDT = AddCurrentTimeDateStampToString(dirname, false);
    cout << dirnamePDT << endl;

    dirnamePDT = AddCurrentTimeDateStampToString(dirname, true);
    cout << dirnamePDT << endl;
}

TEST_F(TestFileUtils, TestAddIndexToFilename)
{
    string fn("TstFilename");
    string filenamePIdx = AddIndexToFilename(fn, 23, 4, "xyz");
    cout << filenamePIdx << endl;

    filenamePIdx = AddIndexToFilename(fn, 7, 3, ".dat");
    cout << filenamePIdx << endl;
}

TEST_F(TestFileUtils, TestAddFileExtention)
{
    string fn1("TstFilename");
    string fn1wext = AddOrReplaceFileExtention(fn1, "ini");
    cout << fn1wext << endl;

    string fn2("TstFilename.data");
    string fn2wext = AddOrReplaceFileExtention(fn2, "ini");
    cout << fn2wext << endl;

    string fn3("TextFilename.txt");
    string fn3wext = AddOrReplaceFileExtention(fn3, ".txt");
    cout << fn3wext << endl;

}
