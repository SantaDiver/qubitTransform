/*
Program usage
./prog <number of processes> <n, where 2^n is vector length> 

or
./prog <number of processes> <n, where 2^n is vector length> -adamar <k number of kubit> 

The second will use U matrix with Adamar matrix
*/

#include <iostream>
#include <omp.h>  
#include <math.h>
#include <vector>
#include <complex>
#include <string.h>
#include <ctime>

typedef std::complex<float> complexd;
typedef unsigned int uint;

using std::cout;
using std::endl;
using std::vector;

#define tsrandom(min, max, seed) min + static_cast <float> (rand_r(&seed)) / ( static_cast <float> (RAND_MAX/(max-min)));

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 5) throw 1;
    if (argc == 5 && strcmp(argv[3], "-adamar")) throw 2;
    
    clock_t begin = clock();
    
    uint n = atoi(argv[2]);
    if (n < 1 || n > 100) throw 3;
    
    long long int vectorLength = pow(2, n);
    
    vector<complexd> oldState(vectorLength);
    vector<complexd> newState(vectorLength);
    
    unsigned seed;
    srand(time(0));
    const int C1 = rand()%25000;
    const int C2 = rand()%100;
    
    const int LO = -10;
    const int HI = 10;
    
    uint ntr = atoi(argv[1]);
    if (ntr < 1 || ntr > 1024) throw 4;
    
    #pragma omp parallel num_threads(ntr) private(seed) default(shared)
    {
        seed = C1 + C2*omp_get_thread_num();
        #pragma omp for
        for (uint i=0; i<vectorLength; ++i)
        {
            float a = tsrandom(LO, HI, seed);
            float b = tsrandom(LO, HI, seed);
            oldState[i] = complexd(a, b);
        }
    }
    
    seed = rand()%25000;
    complexd u[2][2];
    uint k;
    if (argc == 3)
    {
        for (uint i=0; i<2; ++i)
        {
            for (uint j=0; j<2; ++j)
            {
                float a = tsrandom(LO, HI, seed);
                float b = tsrandom(LO, HI, seed);
                u[i][j] = complexd(a, b);
            }
        }
        
        k = rand()%n + 1;
    }
    else
    {
        float c = 1 / sqrt(2);
        u[0][0] = complexd(c, 0);
        u[0][1] = complexd(c, 0);
        u[1][0] = complexd(c, 0);
        u[1][1] = complexd(-c, 0);
        
        k = atoi(argv[4]);
        if (k > n) throw 5;
    }
    
    #pragma omp parallel num_threads(ntr) default(shared)
    {
        #pragma omp for
        for (uint i=0; i<vectorLength; ++i)
        {
            int ik = (i >> (n-k)) & 2 >> 1;
            newState[i] = u[ik][0]*oldState[i - pow(2, k-1)] + u[ik][1]*oldState[i]; 
        }
    }
    
    // cout << "Old state" << endl;
    // for (int i=0; i<vectorLength; ++i)
    // {
    //     cout << oldState[i] << endl;
    // }
    // cout << endl;
    
    // cout << "New state" << endl;
    // for (int i=0; i<vectorLength; ++i)
    // {
    //     cout << newState[i] << endl;
    // }
    
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << elapsed_secs << endl;
}