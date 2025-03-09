
#include <iostream>
#include <string>

#include "TestCudaSmVecs.h"

using namespace std;

//Setup and run tests on the Cuda Common Library items.
int main(int argc, char **argv)
{
    int numberOfErrors = 0;
    string userresponse;
    cout << "***************************************************" << std::endl;
    cout << "*              Test ConfigData                    *" << std::endl;
    cout << "***************************************************" << std::endl;
    cout << std::endl;

    //Setup and call test functions here.
    TestCudaSmVecs tstSmVecsObj;
    numberOfErrors += tstSmVecsObj.TestSmVecs();

    cout << "End of Cuda Common Lib Tests, Number of Errors = " << numberOfErrors << std::endl;
    cout << "Hit return to complete.";
    cin >> userresponse;
    return numberOfErrors;
}