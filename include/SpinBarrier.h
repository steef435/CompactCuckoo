#ifndef GPUCODE
#include <atomic>
#include <Windows.h>

//Based on: https://stackoverflow.com/questions/8115267/writing-a-spinning-thread-barrier-using-c11-atomics
class SpinBarrier
{
public:
    SpinBarrier(unsigned int n) : n_(n), nwait_(0), step_(0) {}

    void Wait()
    {
        unsigned int step = step_.load();

        if (nwait_.fetch_add(1) == n_ - 1)
        {
            nwait_.store(0);
            step_.fetch_add(1);
            return;
        }
        else
        {
            while (step_.load() == step) {
                YieldProcessor();
            }
                ;
            return;
        }
    }

    void signalThreadStop() {
        if (nwait_== n_.fetch_sub(1) - 1)
        {
            nwait_.store(0);
            step_.fetch_add(1);
            return;
        }
    }

protected:
    /* Number of synchronized threads. */
    std::atomic<unsigned int> n_;

    /* Number of threads currently spinning.  */
    std::atomic<unsigned int> nwait_;

    /* Number of barrier syncronizations completed so far*/
    std::atomic<unsigned int> step_;
};
#endif
