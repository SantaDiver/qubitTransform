/*

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
#include <stdlib.h>
#include "mpi.h"

typedef std::complex<float> complexd;
typedef unsigned int uint;

using std::cout;
using std::endl;
using std::vector;

#define tsrandom(min, max, seed) min + static_cast <float> (rand_r(&seed)) / \
    ( static_cast <float> (RAND_MAX/(max-min)));
    
int rank = 0, comm_size;
    
void countCubit(vector<complexd> &vec, vector<complexd> &result, uint n, uint k)
{
    complexd u[2][2];
    float c = 1 / sqrt(2);
    u[0][0] = complexd(c, 0);
    u[0][1] = complexd(c, 0);
    u[1][0] = complexd(c, 0);
    u[1][1] = complexd(-c, 0);
    
    uint procNum = ( ( 1 << (n-k) ) / vec.size() );
    
    MPI_Sendrecv
    (
        vec.data(), vec.size(), MPI::COMPLEX, procNum ^ rank, 0,
	    result.data(), result.size(), MPI::COMPLEX, procNum ^ rank, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
    
    for (uint i=0; i<vec.size(); ++i)
    {
        if (!(rank & procNum))
            result[i] = u[0][0] * vec[i] + u[0][1] * result[i];
        else
            result[i] = u[1][0] * vec[i] + u[1][1] * result[i];
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (argc != 3) return -1;
    
    uint n = atoi(argv[1]);
    if (n < 1 || n > 100) return -1;
    
    uint k = atoi(argv[2]);
    if (k > n) return -1;
    
    double startTime = MPI_Wtime();
    
    long long int vectorLength = pow(2, n) / comm_size;
    
    vector<complexd> oldState(vectorLength);
    vector<complexd> newState(vectorLength);
    
    const int LO = -10;
    const int HI = 10;
    
    uint seed = (time(0)+rank*607)%8345;
    srand((time(0)+123*rank)%1039);
    
    for (uint i = 0; i < oldState.size(); ++i){
        float a = tsrandom(LO, HI, seed);
        float b = tsrandom(LO, HI, seed);
        oldState[i] = complexd(a, b);
    }
    
    
    countCubit(oldState, newState, n, k);

    double workTime = MPI_Wtime()-startTime;
    
    double maxtime;
    MPI_Reduce(&workTime, &maxtime, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank==0) cout << maxtime << endl;
    
    MPI_Finalize();
    return 0;
}