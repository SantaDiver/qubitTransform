#include <iostream>
#include <cmath>
#include <fstream>
#include <complex>
#include <algorithm>
#include <vector>
#include <string>
#include "mpi.h"
#include <omp.h>


using namespace std;

typedef unsigned int uint;
typedef std::complex<double> complexd;
typedef std::vector< std::vector<complexd> > cmatrix;

int rank = 0, comm_size;
uint glNumberOfThreads;

double tsrandom(double min, double max, uint seed)
{
    return min + static_cast <double> (rand_r(&seed)) / 
        ( static_cast <double> (RAND_MAX/(max-min)));
}

uint countIk(uint i, uint leap1, uint leap2, uint k1formLeft, uint k2formLeft)
{
    uint ik1 = (i & leap1) >> k1formLeft;
    uint ik2 = (i & leap2) >> k2formLeft;
    
    return (ik1 << 1) + ik2;
}

void countDoubleCubit(vector<complexd>& a, vector<complexd>& b, uint sizeOfProcessPart, 
    uint n, uint k1, uint k2, cmatrix H)
{
    uint k1formLeft = n - k1;
    uint leap1 = 1 << k1formLeft;
    uint k2formLeft = n - k2;
    uint leap2 = 1 << k2formLeft;

    if (leap1 < sizeOfProcessPart && leap2 < sizeOfProcessPart)
    {
        #pragma omp parallel for num_threads(glNumberOfThreads)
        for (uint i = 0; i < sizeOfProcessPart; ++i)
        {
            uint i00 = i & ~leap1 & ~leap2;
            uint i01 = i & ~leap1 | leap2;
            uint i10 = (i | leap1) & ~leap2;
            uint i11 = i | leap1 | leap2;

            uint ik = countIk(i, leap1, leap2, k1formLeft, k2formLeft);
            
            b[i] = H[ik][0] * a[i00] + H[ik][1] * a[i01] +
            H[ik][2] * a[i10] + H[ik][3] * a[i11];
        }
    } 
    else if (leap1 < sizeOfProcessPart)
    {
        uint procLeap = leap1 / sizeOfProcessPart;
        vector<complexd> tmp(sizeOfProcessPart);

        MPI_Sendrecv
        (
            a.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap^rank, 0,
            tmp.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap^rank, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        #pragma omp parallel for num_threads(glNumberOfThreads)
        for (uint i = 0; i < sizeOfProcessPart; ++i)
        {
            uint i00 = i & ~leap1 & ~leap2;
            uint i01 = i & ~leap1;
            uint i10 = (i | leap1) & ~leap2;
            uint i11 = i | leap1;
            
            uint ik = countIk(i, leap1, leap2, k1formLeft, k2formLeft);

            b[i] = H[ik][0] * a[i00] + H[ik][1] * tmp[i01] +
            H[ik][2] * a[i10] + H[ik][3] * tmp[i11];

        }
    } 
    else if (leap2 < sizeOfProcessPart)
    {
        uint procLeap = leap2 / sizeOfProcessPart;
        vector<complexd> tmp(sizeOfProcessPart);

        MPI_Sendrecv
        (
            a.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap^rank, 0,
            tmp.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap^rank, 0, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        #pragma omp parallel for num_threads(glNumberOfThreads)
        for (uint i = 0; i < sizeOfProcessPart; ++i)
        {
            uint i00 = i & ~leap1 & ~leap2;
            uint i01 = i & ~leap1 | leap2;
            uint i10 = i & ~leap2;
            uint i11 = i | leap2;
            
            uint ik = countIk(i, leap1, leap2, k1formLeft, k2formLeft);

            b[i] = H[ik][0] * a[i00] + H[ik][1] * a[i01] +
            H[ik][2] * tmp[i10] + H[ik][3] * tmp[i11];
        }
    } 
    else
    {
        uint procLeap1 = leap1 / sizeOfProcessPart;
        uint procLeap2 = leap2 / sizeOfProcessPart;
        vector<complexd> tmp1(sizeOfProcessPart),tmp2(sizeOfProcessPart),tmp3(sizeOfProcessPart);

        MPI_Sendrecv
        (
            a.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap1^rank, 0,
            tmp1.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap1^rank, 0, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        MPI_Sendrecv
        (
            a.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap2^rank, 0,
            tmp2.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap2^rank, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        MPI_Sendrecv
        (
            a.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap1^procLeap2^rank, 0,
            tmp3.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap1^procLeap2^rank, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        #pragma omp parallel for num_threads(glNumberOfThreads)
        for (uint i = 0; i < sizeOfProcessPart; ++i)
        {
            uint i00 = i & ~leap1 & ~leap2;
            uint i01 = i & ~leap1;
            uint i10 = i & ~leap2;
            uint i11 = i;
            
            uint ik = countIk(i, leap1, leap2, k1formLeft, k2formLeft);
            
            b[i] = H[ik][0] * a[i00] + H[ik][1] * tmp2[i01] +
            H[ik][2] * tmp1[i10] + H[ik][3] * tmp3[i11];

        }
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    try
    {
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (argc != 5) throw string("Wrong arguments");

        uint n = atoi(argv[1]);
        if (n < 1 || n > 100) throw string("-1");
        
        uint k1 = atoi(argv[2]);
        if (k1 < 1 || k1 > n) throw string("-2");
        
        uint k2 = atoi(argv[3]);
        if (k2 < 1 || k2 > n) throw string("-2");
        
        glNumberOfThreads = atoi(argv[4]);
        if (glNumberOfThreads < 1 || glNumberOfThreads > 100) throw string("-3");

        uint sizeOfVector = pow(2, n);
        if (sizeOfVector % comm_size != 0 || comm_size > sizeOfVector)
            throw string("Wrong number of processors");

        uint sizeOfProcessPart = sizeOfVector / comm_size;
        
        double startTime = MPI_Wtime();

        vector<complexd> initialVec(sizeOfProcessPart);
        vector<complexd> endingVec(sizeOfProcessPart);
        
        int seed = time(0) + 1713*rank;
        srand(seed + 2222 % rank);
        
        const int LO = -1000;
        const int HI = 1000;

        double sumOfModsSquares = 0;
        
        #pragma omp parallel for num_threads(glNumberOfThreads)
        for (uint i=0; i < sizeOfProcessPart; ++i)
        {
            double a = tsrandom(LO, HI, seed);
            double b = tsrandom(LO, HI, seed);
            initialVec[i] = complexd(a, b);
            
            sumOfModsSquares += pow(abs(initialVec[i]), 2);
        }
        
        double sumOfAllModsSquares = 0;
        MPI_Allreduce(&sumOfAllModsSquares, &sumOfModsSquares, 1, 
            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        double norm = sqrt(sumOfAllModsSquares);
        for (uint i = 0; i < sizeOfProcessPart; ++i)
        {
            initialVec[i] /= norm;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        cmatrix H = cmatrix(4, vector<complexd> (4, 0));
        H[0][0] = 1;
        H[1][1] = 1;
        H[2][3] = 1;
        H[3][2] = 1;

        countDoubleCubit(initialVec, endingVec, sizeOfProcessPart, n, k1, k2, H);

        double workTime = MPI_Wtime()-startTime;

        double maxtime;
        MPI_Reduce(&workTime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank==0) cout << "Time " << maxtime << endl;
    }
    catch (const string& e) {
        if (rank == 0) cerr << e << endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    catch (const MPI::Exception& e) {
        if (rank == 0) cerr << "MPI Exception thrown" << endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    MPI_Finalize();
    return 0;
}


