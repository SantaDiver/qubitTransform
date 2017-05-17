/*

Made by SNM

*/
#define matrix2x2 cmatrix(2, vector<complexd> (2, 0))

#include <iostream>
#include <omp.h>
#include <math.h>
#include <vector>
#include <complex>
#include <string.h>
#include <ctime>
#include <stdlib.h>
#include "mpi.h"

typedef std::complex<double> complexd;
typedef unsigned int uint;
typedef std::vector< std::vector<complexd> > cmatrix;

using namespace std;

int rank = 0, comm_size;

uint glNumberOfThreads=1;

uint getBit(uint a, uint k, uint n)
{ 
    if ( ( a & ( 1 << (n-k) ) ) == 0 ) return 0;
    else return 1;
}

uint withBit(uint a, uint k, uint bit, uint n) 
{ 
    if (bit==1) {
        return a | (bit << (n - k)); 
    }
    else {
        return a & ( ~(1 << (n - k) ) );
    }
}

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

    if (leap1 < sizeOfProcessPart && leap2 < sizeOfProcessPart) {
        #pragma omp parallel for num_threads(glNumberOfThreads)
        for (uint i = 0; i < sizeOfProcessPart; ++i) {
            uint i00 = i & ~leap1 & ~leap2;
            uint i01 = i & ~leap1 | leap2;
            uint i10 = (i | leap1) & ~leap2;
            uint i11 = i | leap1 | leap2;

            uint ik = countIk(i, leap1, leap2, k1formLeft, k2formLeft);
            
            b[i] = H[ik][0] * a[i00] + H[ik][1] * a[i01] +
            H[ik][2] * a[i10] + H[ik][3] * a[i11];
        }
    } 
    else if (leap1 < sizeOfProcessPart) {
        uint procLeap = leap1 / sizeOfProcessPart;
        vector<complexd> tmp(sizeOfProcessPart);

        MPI_Sendrecv (
            a.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap^rank, 0,
            tmp.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap^rank, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        #pragma omp parallel for num_threads(glNumberOfThreads)
        for (uint i = 0; i < sizeOfProcessPart; ++i) {
            uint i00 = i & ~leap1 & ~leap2;
            uint i01 = i & ~leap1;
            uint i10 = (i | leap1) & ~leap2;
            uint i11 = i | leap1;
            
            uint ik = countIk(i, leap1, leap2, k1formLeft, k2formLeft);

            b[i] = H[ik][0] * a[i00] + H[ik][1] * tmp[i01] +
            H[ik][2] * a[i10] + H[ik][3] * tmp[i11];

        }
    } 
    else if (leap2 < sizeOfProcessPart) {
        uint procLeap = leap2 / sizeOfProcessPart;
        vector<complexd> tmp(sizeOfProcessPart);

        MPI_Sendrecv (
            a.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap^rank, 0,
            tmp.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap^rank, 0, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        #pragma omp parallel for num_threads(glNumberOfThreads)
        for (uint i = 0; i < sizeOfProcessPart; ++i) {
            uint i00 = i & ~leap1 & ~leap2;
            uint i01 = i & ~leap1 | leap2;
            uint i10 = i & ~leap2;
            uint i11 = i | leap2;
            
            uint ik = countIk(i, leap1, leap2, k1formLeft, k2formLeft);

            b[i] = H[ik][0] * a[i00] + H[ik][1] * a[i01] +
            H[ik][2] * tmp[i10] + H[ik][3] * tmp[i11];
        }
    } 
    else {
        uint procLeap1 = leap1 / sizeOfProcessPart;
        uint procLeap2 = leap2 / sizeOfProcessPart;
        vector<complexd> tmp1(sizeOfProcessPart),tmp2(sizeOfProcessPart),tmp3(sizeOfProcessPart);

        MPI_Sendrecv (
            a.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap1^rank, 0,
            tmp1.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap1^rank, 0, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        MPI_Sendrecv (
            a.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap2^rank, 0,
            tmp2.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap2^rank, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        MPI_Sendrecv (
            a.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap1^procLeap2^rank, 0,
            tmp3.data(), sizeOfProcessPart, MPI::COMPLEX, procLeap1^procLeap2^rank, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        #pragma omp parallel for num_threads(glNumberOfThreads)
        for (uint i = 0; i < sizeOfProcessPart; ++i) {
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

void countCubit(vector <complexd> &vec, vector <complexd> &res, int k, uint n, cmatrix u)
{
    uint block_size = vec.size();
    uint block_shift = (1 << (n - k)) / block_size;

    if (block_shift == 0) {
        #pragma omp parallel for num_threads(glNumberOfThreads)
        for (uint i=0; i < block_size; ++i) {
            uint Ik = getBit(i, k, n);
            res[i] = u[Ik][0] * vec[withBit(i, k, 0, n)] + u[Ik][1] * vec[withBit(i, k, 1, n)];
        }
    }
    else {
        vector<complexd> tmp(block_size);
        MPI_Sendrecv(
            vec.data(), block_size, MPI::COMPLEX, rank ^ block_shift, 0, tmp.data(),
            block_size, MPI::COMPLEX, rank ^ block_shift, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        #pragma omp parallel for num_threads(glNumberOfThreads)
        for (uint i=0; i < block_size; ++i)  {
            if (rank < (rank ^ block_shift))
                res[i] = u[0][0] * vec[i] + u[0][1] * tmp[i];
            else
                res[i] = u[1][0] * tmp[i] + u[1][1] * vec[i];
        }
    }
}


cmatrix generateAdamarMatrix()
{
    cmatrix u = matrix2x2;
    double c = 1 / sqrt(2);
    u[0][0] = complexd(c, 0);
    u[0][1] = complexd(c, 0);
    u[1][0] = complexd(c, 0);
    u[1][1] = complexd(-c, 0);

    return u;
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    
    try {
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (argc != 5) throw string("Wrong arguments");

        uint n = atoi(argv[1]);
        if (n < 1 || n > 100) throw string("-1");
        
        glNumberOfThreads = atoi(argv[2]);
        if (glNumberOfThreads < 1 || glNumberOfThreads > 100) throw string("-3");

        uint sizeOfVector = pow(2, n);
        if (sizeOfVector % comm_size != 0 || comm_size > sizeOfVector)
            throw string("Wrong number of processors");

        uint sizeOfProcessPart = sizeOfVector / comm_size;

        vector<complexd> initialVec(sizeOfProcessPart);
        vector<complexd> endingVec(sizeOfProcessPart);
        
        MPI_File fileToReadFrom;
        MPI_Offset offset = rank*sizeOfProcessPart*sizeof(complexd);

        MPI_File_open(
            MPI_COMM_WORLD, 
            argv[3], 
            MPI_MODE_RDONLY, 
            MPI_INFO_NULL, 
            &fileToReadFrom
        );
        
        MPI_File_read_at(
            fileToReadFrom, 
            offset, 
            initialVec.data(), 
            sizeOfProcessPart, 
            MPI_DOUBLE_COMPLEX, 
            MPI_STATUS_IGNORE
        );
        
        MPI_File_close(&fileToReadFrom);

        MPI_Barrier(MPI_COMM_WORLD);
        
        double startTime = MPI_Wtime();
        
        cmatrix adamarMatrix = generateAdamarMatrix();

        cmatrix H = cmatrix(4, vector<complexd> (4, 0));
        H[0][0] = 1;
        H[1][1] = 1;
        H[2][2] = 1;
        
        for(uint i=1; i<=n; ++i) {
            for(uint j=1; j<i; ++j) {
                H[3][3] = exp(complexd(0,1) * M_PI / pow(2, j));
                countDoubleCubit(initialVec, endingVec, sizeOfProcessPart, n, i, j, H);
                initialVec.swap(endingVec);
            }
            countCubit(initialVec, endingVec, i, n, adamarMatrix);
            initialVec.swap(endingVec);
        }
        
        H = cmatrix(4, vector<complexd> (4, 0));
        H[0][0] = 1;
        H[1][2] = 1;
        H[2][1] = 1;
        H[3][3] = 1;
    
        for(uint i=1; i<n/2; ++i) {
            countDoubleCubit(initialVec, endingVec, sizeOfProcessPart, n, i, n-i+1, H);
            initialVec.swap(endingVec);
        }
        
        MPI_File fileToWrite;

        MPI_File_open(
            MPI_COMM_WORLD, 
            argv[4], 
            MPI_MODE_CREATE|MPI_MODE_WRONLY, 
            MPI_INFO_NULL, 
            &fileToWrite
        );
        
        MPI_File_write_at(
            fileToWrite, 
            offset, 
            initialVec.data(), 
            sizeOfProcessPart, 
            MPI_DOUBLE_COMPLEX, 
            MPI_STATUS_IGNORE
        );
        
        MPI_File_close(&fileToWrite);
        
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
