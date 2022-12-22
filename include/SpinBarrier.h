#ifndef GPUCODE
#include <atomic>
#include <Windows.h>

//Based on: https://stackoverflow.com/questions/8115267/writing-a-spinning-thread-barrier-using-c11-atomics
/*
class SpinBarrier
{
public:
    SpinBarrier(unsigned int n) : n_(n), nwait_(0), step_(0) {}

    void Wait()
    {
        printf("%i: \tEnter Wait\n", getThreadID());
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
            printf("%i: \tExit Wait\n", getThreadID());
            return;
        }
    }

    void signalThreadStop() {
        printf("%i: \tThread Stop\n", getThreadID());
        if (nwait_== n_.fetch_sub(1) - 1)
        {
            nwait_.store(0);
            step_.fetch_add(1);
            return;
        }
        printf("%i: \tThread Stop Done Wait\n", getThreadID());
    }

protected:
    // Number of synchronized threads.
    std::atomic<unsigned int> n_;

    // Number of threads currently spinning.  
    std::atomic<unsigned int> nwait_;

    // Number of barrier syncronizations completed so far
    std::atomic<unsigned int> step_;
};
*/

struct SpinBarrier {
    std::atomic<unsigned int> count;
    std::atomic<unsigned int> spaces;
    std::atomic<unsigned int> generation;
    SpinBarrier(unsigned int count_) :
        count(count_), spaces(count_), generation(0)
    {}
    void Wait() {
        unsigned const int my_generation = generation;
        //printf("Wait: g:%i s:%i\n", generation, spaces);
        if (!--spaces) {
            spaces.store(count);
            ++generation;
        }
        else {
            while (generation == my_generation);
        }
    }
    void signalThreadStop() {
        //printf("SignalThreadStop: g:%i s:%i\n", generation, spaces);
        count--;
        if (!--spaces) {
            spaces.store(count);
            ++generation;
        }
    }
};
#endif
