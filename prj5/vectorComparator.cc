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

void checkResult(vector<complexd> &initialVec, vector<complexd> &endingVec)
{
    uint n = initialVec.size();
    complexd coef = 2 * M_PI / n;
    complexd w = exp(complexd(0,1) * coef);
    
    vector<complexd> endingVector(initialVec.size());
    for(uint i=0; i<initialVec.size(); ++i) {
        for(uint j=0; j<initialVec.size(); ++j)
        {
            endingVector[i] = initialVec[i] * pow(w, j*i);
        }
    }
    
    for(uint i=0; i<initialVec.size(); ++i) {
        cout << endingVec[i] << " " << endingVec[i] << endl;
    }
}


int main(int argc, char** argv)
{
    try {
        if (argc != 3) throw string("Wrong arguments");
        
        FILE* f = 0;
        f = fopen(argv[1],"rb");
        if (f == 0) throw string("No such file!");
        
        vector<complexd> initialVec;
        complexd tmp;
        while (fread(&tmp,sizeof(complexd),1,f) != 0) initialVec.push_back(tmp);
        fclose(f);
        
        f = 0;
        f = fopen(argv[2],"rb");
        if (f == 0) throw string("No such file!");
        
        vector<complexd> endingVec;
        while (fread(&tmp,sizeof(complexd),1,f) != 0) endingVec.push_back(tmp);
        fclose(f);
        
        if (initialVec.size() != endingVec.size()) throw string("Vecs are different size!");
        
        checkResult(initialVec, endingVec);
    }
    catch (const string& e) {
        cerr << e << endl;
        return -1;
    }
    
    return 0;
}
