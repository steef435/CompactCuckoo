#include <condition_variable>
#include <thread>

//Based on: https://stackoverflow.com/a/27118537/11068792
class Barrier {
public:
    Barrier(){}

    Barrier(std::size_t iCount){
        mThreshold = iCount;
        mCount = 0;
        mGeneration = 0;
    }

    //Start Waiting
    void Wait() {
        std::unique_lock<std::mutex> lLock{ mMutex };
        auto lGen = mGeneration;
        if (++mCount >= mThreshold) {
            mGeneration++;
            mCount = 0;
            mCond.notify_all();
        }
        else {
            mCond.wait(lLock, [this, lGen] { return lGen != mGeneration; });
        }
    }

    //Method for when a thread prematurely stops
    void signalThreadStop() {
        std::unique_lock<std::mutex> lLock{ mMutex };
        if (mCount >= --mThreshold) {
            mGeneration++;
            mCount = 0;
            mCond.notify_all();
        }
    }

private:
    std::mutex mMutex;
    std::condition_variable mCond;
    std::size_t mThreshold;
    std::size_t mCount;
    std::size_t mGeneration;
};