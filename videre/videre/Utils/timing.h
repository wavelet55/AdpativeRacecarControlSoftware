/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Dec. 2016
 *
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 * Note: Use the SystemTimeClock from Rabit for all clock/timestamp
 * purposes.
 *
 * **************************************************************/


#ifndef VIDERE_TIMING
#define VIDERE_TIMING

#include <chrono>

namespace videre
{

    class Timing
    {

    public:

        Timing()
        {
            epoch_ = std::chrono::high_resolution_clock::now();
            auto n = std::chrono::high_resolution_clock::now();
            delta_time_ = std::chrono::duration_cast<std::chrono::microseconds>(n - epoch_);
        }

        void Now()
        {
            epoch_ = std::chrono::high_resolution_clock::now();
        }

        double dt_micro()
        {
            std::chrono::duration<double, std::micro> r = delta_time_;
            return r.count();
        }

        double dt_milli()
        {
            std::chrono::duration<double, std::milli> r = delta_time_;
            return r.count();
        }

        double dt_sec()
        {
            std::chrono::duration<double, std::ratio<1, 1> > r = delta_time_;
            return r.count();
        }

        double fps()
        {
            return 1.0 / dt_sec();
        }

        unsigned long elapsed_micro()
        {
            auto n = std::chrono::high_resolution_clock::now();
            std::chrono::duration<unsigned long, std::micro> r = std::chrono::duration_cast<std::chrono::microseconds>(
                    n - epoch_);
            return r.count();
        }

        unsigned long elapsed_milli()
        {
            auto n = std::chrono::high_resolution_clock::now();
            std::chrono::duration<unsigned long, std::milli> r = std::chrono::duration_cast<std::chrono::milliseconds>(
                    n - epoch_);
            return r.count();
        }

        unsigned long elapsed_sec()
        {
            auto n = std::chrono::high_resolution_clock::now();
            std::chrono::duration<unsigned long, std::ratio<1, 1> > r = std::chrono::duration_cast<std::chrono::seconds>(
                    n - epoch_);
            return r.count();
        }

        void set_delta_time(std::chrono::microseconds dt)
        {
            delta_time_ = dt;
        }

    private:
        std::chrono::microseconds delta_time_;
        std::chrono::high_resolution_clock::time_point epoch_;
    };


}


#endif //VIDERE_TIMING
