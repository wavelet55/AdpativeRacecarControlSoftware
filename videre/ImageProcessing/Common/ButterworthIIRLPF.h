/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: March. 2016
 *
     Butterworth filter analysis was obtained from:  http://www.kwon3d.com/theory/filtering/fil.html\n",
     The artical gives coeficient calculations for 2nd, 4th, 6th and so on order filters.\n",
     A 2nd Order IIR Low Pass filter has the form:\n",
       H(z) = (a0 + a1*z^-1 + a2*z^-2)  /  (1 + b1*z^-1 + b2*z^-2)\n",

       The sample frequency is:  fs (Hz)\n",
       The cutoff frequency is:  fc (Hz)\n",
       The warped normalized Freq: fw = tan(pi * fc / fs)\n",
  *******************************************************************/


#ifndef VIDERE_DEV_BUTTERWORTHIIRLPF_H
#define VIDERE_DEV_BUTTERWORTHIIRLPF_H

#include <math.h>
#include <boost/math/constants/constants.hpp>
#include "XYZCoord_t.h"
#include "MathUtils.h"

namespace MathLibsNS
{

    class Butterworth2nOrderIIRLPF
    {
        double a0, a1, a2;
        double b1, b2;
        double xs[2];
        double ys[2];


    public:
        Butterworth2nOrderIIRLPF() {}

        //This must be called before using the filter.
        void generateButterworthLPFCoeficients(double Freq_cutoffHz, double Freq_sampleHz, int k, int N);

        //Set the filter initial states and output.
        void setInitialStateOrFilterOuput(double yout);

        //Compute the next filter output given the input.
        //If the filter is used in a cascade... set updateState to false.
        double fx(double xin, bool updateState = true);

        XYZCoord_t updateState(XYZCoord_t &state);

    };


    class ButterworthIIRLPF
    {
        const int MaxNo2ndOrderFilters = 5;
        Butterworth2nOrderIIRLPF BwLPFs[5];

        int _numberCascadeFilters = 1;

    public:
        //These values are for reference only.
        //If OrderN = 0... the filter is by-passed.
        int OrderN = 2;
        double FreqSampleHz = 100.0;
        double FreqCutoffHz = 5.0;


        ButterworthIIRLPF(double N = 2, double FsampleHz = 100.0, double FcutoffHz = 10.0);


        //This must be called before using the filter.
        //The Filter order must be even:  2, 4, 6..
        void createButterworthFilter(int N, double Freq_cutoffHz, double Freq_sampleHz);

        //Set the filter initial states and output.
        void setInitialStateOrFilterOuput(double yout);

        //Compute the next filter output given the input.
        double fx(double xin);

    };


}

#endif //VIDERE_DEV_BUTTERWORTHIIRLPF_H
