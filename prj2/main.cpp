#include <iostream>
#include <fstream>
#include "Matrix.h"
#include <algorithm>
#include <vector>
#include "mpi.h"
#include <cmath>

using namespace std;
using namespace Matrix_ns;

void cubit(vector<complex<float> >& a, vector<complex<float> >& b, int loc_size, int N, int K){
    static Matrix<complex<float> > H(2, 2);
    H(0, 0) = 1 / std::sqrt(2);
    H(0, 1) = 1 / std::sqrt(2);
    H(1, 0) = 1 / std::sqrt(2);
    H(1, 1) = -1 / std::sqrt(2);

    int P = N - K;
    int stride = 1 << P;

    if (stride < loc_size) {
        // All elements are here in a
        for (int i = 0; i < loc_size; ++i){
            int j_1 = i & ~stride;
            int j_2 = i | stride;
            int u_i = !((i & stride) == 0);
            b[i] = H(u_i, 0) * a[j_1] + H(u_i, 1) * a[j_2];

        }
    } else {
	// need to exchange "a" with another processor
        int proc_stride = stride / loc_size;

        MPI_Sendrecv(a.data(),
                 loc_size,
                 MPI::COMPLEX,
                 proc_stride^rank,
                 0,
		 b.data(),
		 loc_size,
		 MPI::COMPLEX,
		 proc_stride^rank,
		 0,
                 MPI_COMM_WORLD,
		 MPI_STATUS_IGNORE);

        for (int i = 0; i < loc_size; ++i) {
            if (!(rank & proc_stride))
                b[i] = H(0, 0) * a[i] + H(0, 1) * b[i];
            else
                b[i] = -(H(1, 0) * a[i] + H(1, 1) * b[i]);
        }
    }
}


//#define DEBUG
#undef DEBUG

int rank = 0, comm_size;

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    uint N = atoi(argv[1]), uint K = atoi(argv[2]);

    uint vec_size = (uint) std::pow((float) 2, (int) N);
    if (vec_size % comm_size != 0 || (uint) comm_size > vec_size) {
        cout << "Wrong number of processors";
        return 0;
    }

    uint loc_size = vec_size / comm_size;
    int seed = 0;
    double start_time, end_time, time_diff_comp = 0;
    double all_times_comp[comm_size];

    vector<complex<float> > a(loc_size), b(loc_size);

    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    srand((seed == 0)? time(0) + rank : seed + rank);

    for (uint i = 0; i < loc_size; ++i){
        a[i] = complex<float>(std::rand()/( RAND_MAX / 100. ), std::rand() / ( RAND_MAX / 100. );
    }


    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    cubit(a, b, loc_size, N, K);

    end_time = MPI_Wtime();
    time_diff_comp = end_time - start_time;


    MPI_Gather(&time_diff_comp, 1, MPI::DOUBLE, all_times_comp, 1, MPI::DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << N << "    " << K << "    " << comm_size << "    Max_comp " << *std::max_element(&all_times_comp[0], &all_times_comp[comm_size]) << endl;
    }

    MPI_Finalize();
    return 0;
}
