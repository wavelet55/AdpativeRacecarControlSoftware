/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: March. 2016
 *
  *******************************************************************/

#include "ButterworthIIRLPF.h"

namespace MathLibsNS
{

    //This must be called before using the filter.
    void Butterworth2nOrderIIRLPF::generateButterworthLPFCoeficients(double Freq_cutoffHz, double Freq_sampleHz, int k, int N)
    {
        double PI = boost::math::constants::pi<double>();
        //Ensure N is even
        N = N >> 1;
        k = k < 0 ? 0 : k >= N ? N - 1 : k;
        N = N << 1;
        N = N < 2 ? 2 : N > 10 ? 10 : N;
        Freq_sampleHz = Freq_sampleHz < 0.001 ? 0.001 : Freq_sampleHz;
        Freq_cutoffHz = Freq_cutoffHz < 1e-6 ? 1e-6 : Freq_cutoffHz;
        Freq_cutoffHz = Freq_cutoffHz > 0.25 * Freq_sampleHz ? 0.25 * Freq_sampleHz : Freq_cutoffHz;

        double fw = tan(PI * Freq_cutoffHz / Freq_sampleHz);
        double fwsq = fw * fw;
        double c0 = 2.0 * cos(PI * (2.0 * (double)k + 1)/(2.0 * (double)N));
        double c1 = 1.0 + c0 * fw + fwsq;

        a0 = fwsq / c1;
        a1 = 2.0 * a0;
        a2 = a0;
        b1 = 2.0 * (fwsq - 1.0) / c1;
        b2 = (1.0 - c0 * fw + fwsq) / c1;

        setInitialStateOrFilterOuput(0);
    }

    //Set the filter initial states and output.
    void Butterworth2nOrderIIRLPF::setInitialStateOrFilterOuput(double yout)
    {
        for(int i = 0; i < 2; i++)
        {
            xs[i] = yout;
            ys[i] = yout;
        }
    }

    //Compute the next filter output given the input.
    double Butterworth2nOrderIIRLPF::fx(double xin, bool updateState)
    {
        double yout = a0 * xin + a1 * xs[0] + a2 * xs[1];
        yout += -b1 * ys[0] - b2 * ys[1];
        if(updateState)
        {
            xs[1] = xs[0];
            xs[0] = xin;
            ys[1] = ys[0];
            ys[0] = yout;
        }
        return yout;
    }


    XYZCoord_t Butterworth2nOrderIIRLPF::updateState(XYZCoord_t &state)
    {
        XYZCoord_t oldState(xs[1], ys[1]);
        xs[1] = xs[0];
        xs[0] = state.x;
        ys[1] = ys[0];
        ys[0] = state.y;
        return oldState;
    }


    ButterworthIIRLPF::ButterworthIIRLPF(double N, double FsampleHz, double FcutoffHz)
    {
        createButterworthFilter(N, FcutoffHz, FsampleHz);
    }


    //This must be called before using the filter.
    void ButterworthIIRLPF::createButterworthFilter(int N, double Freq_cutoffHz, double Freq_sampleHz)
    {
        N = N >> 1;
        N = N << 1;
        OrderN = N < 0 ? 0 : N > 2 * MaxNo2ndOrderFilters ? 2 * MaxNo2ndOrderFilters : N;
        _numberCascadeFilters = OrderN >> 1;
        FreqSampleHz = Freq_sampleHz;
        FreqCutoffHz = Freq_cutoffHz;

        for(int i = 0; i < _numberCascadeFilters; i++)
        {
            BwLPFs[i].generateButterworthLPFCoeficients(Freq_cutoffHz, Freq_sampleHz, i, OrderN);
        }
    }

    //Set the filter initial states and output.
    void ButterworthIIRLPF::setInitialStateOrFilterOuput(double yout)
    {
        for(int i = 0; i < _numberCascadeFilters; i++)
        {
            BwLPFs[i].setInitialStateOrFilterOuput(yout);
        }
    }

    //Compute the next filter output given the input.
    double ButterworthIIRLPF::fx(double xin)
    {
        XYZCoord_t state;
        double yout = xin;

        for(int i = 0; i < _numberCascadeFilters; i++)
        {
            yout = BwLPFs[i].fx(yout, false);
        }
        //Update States:
        state.x = xin;
        state.y = yout;
        for(int i = _numberCascadeFilters - 1; i >= 0; i--)
        {
            state = BwLPFs[i].updateState(state);
        }
        return yout;
    }


}