//
// Created by nvidia on 7/17/18.
//

#include "TestCubicSpline.h"
#include "global_defines.h"
#include <memory>
#include <string>
#include <boost/filesystem.hpp>
#include "boost/date_time.hpp"
#include "math.h"
#include "FileUtils.h"
#include "CubicSplineProcessor.h"

using namespace std;
using namespace boost::filesystem;
using namespace videre;
using namespace VidereFileUtils;
using namespace VidereUtils;

bool compareEqualDouble2(double x, double y, double epsolon = 1.0e-6)
{
    bool equal = false;
    if( abs(x - y) < epsolon)
        equal = true;
    return equal;
}

TEST_F(TestCubicSpline, TestReadCubicSplineCoefFile)
{
    string filename = "/home/nvidia/Racecar/CubicSplineFiles/Track_CubicSpline_X.csv";
    CubicSplineProcessor csp;
    bool error = csp.readCoefsFromFile(filename);

    if(!error)
    {
        double y = csp.computeFx(27.38);
        double yp = csp.computeFx1stDir(56.29);
        double ypp = csp.computeFx2ndDir(38.53);
    }
}
