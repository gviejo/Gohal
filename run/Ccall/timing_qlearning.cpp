#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <math.h>
#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include "qlearning_call.cpp"
// #include "bayesian_call.cpp"
// #include "selection_call.cpp"
// #include "mixture_call.cpp"
// #include "fusion_call.cpp"

using namespace std;


int main (int argc, char** argv) {	
    // for (int i = 0; i < argc; ++i) {
    //     std::cout << argv[i] << std::endl;
    // }	
  	double fit [2] = {0.0, 0.0};
  	int N = 156; // fmri
    
    const char* str = argv[1];
  	double alpha = atof(argv[2]);
  	double beta = atof(argv[3]);	
    double sigma = atof(argv[4]);
  	
  	fit[0] = 0.0; fit[1] = 0.0;  
    sferes_call(fit, N, str, alpha, beta, sigma);
  	

 	return 0;
}