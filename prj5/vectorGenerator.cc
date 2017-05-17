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

double tsrandom(double min, double max, uint &seed)
{
    return min + static_cast <double> (rand_r(&seed)) / 
        ( static_cast <double> (RAND_MAX/(max-min)));
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    
    try {
        double startTime = MPI_Wtime();
        
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (argc != 4) throw string("Wrong arguments");

        uint n = atoi(argv[1]);
        if (n < 1 || n > 100) throw string("-1");
        
        glNumberOfThreads = atoi(argv[2]);
        if (glNumberOfThreads < 1 || glNumberOfThreads > 100) throw string("-3");

        uint sizeOfVector = pow(2, n);
        if (sizeOfVector % comm_size != 0 || comm_size > sizeOfVector)
            throw string("Wrong number of processors");

        uint sizeOfProcessPart = sizeOfVector / comm_size;

        vector<complexd> initialVec(sizeOfProcessPart);
        
        uint seed = time(0) + 1713*rank;
        srand(time(0) + 2222 * (rank+1));
        
        const int LO = -1000;
        const int HI = 1000;

        double sumOfModsSquares = 0;
        
        #pragma omp parallel num_threads(glNumberOfThreads) private(seed) default(shared)
        {
            seed += 1111 * omp_get_thread_num(); 
            #pragma omp parallel for
            for (uint i=0; i < sizeOfProcessPart; ++i) {
                double a = tsrandom(LO, HI, seed);
                double b = tsrandom(LO, HI, seed);
                initialVec[i] = complexd(a, b);
                
                #pragma omp atomic
                sumOfModsSquares += pow(abs(initialVec[i]), 2);
            }
        }
        
        double sumOfAllModsSquares = 0;
        MPI_Allreduce(&sumOfModsSquares, &sumOfAllModsSquares, 1, 
            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        double norm = sqrt(sumOfAllModsSquares);
        #pragma omp parallel for num_threads(glNumberOfThreads)
        for (uint i=0; i < sizeOfProcessPart; ++i) {
            initialVec[i] /= norm;
        }
        
        MPI_File fileToWrite;
        MPI_Offset offset = rank*sizeOfProcessPart*sizeof(complexd);

        MPI_File_open(
            MPI_COMM_WORLD, 
            argv[3], 
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
