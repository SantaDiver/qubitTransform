/*

Made by SNM

*/

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <complex>
#include <string.h>
#include <ctime>
#include <stdlib.h>

typedef std::complex<double> complexd;
typedef unsigned int uint;
typedef std::vector< std::vector<complexd> > cmatrix;

using namespace std;

int rank = 0, comm_size;

int main(int argc, char** argv)
{
    try {
        if (argc != 2) throw string("Wrong arguments");
        
        FILE* f = 0;
        f = fopen(argv[1],"rb");
        if (f == 0) throw string("No such file!");
        
        complexd c;
        while (fread(&c,sizeof(complexd),1,f) != 0) cout << c << " ";
        cout << endl;
        fclose(f);
    }
    catch (const string& e) {
        cerr << e << endl;
        return -1;
    }
    
    return 0;
}
