#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <math.h>
// #include "qlearning_call.cpp"
// #include "bayesian_call.cpp"
// #include "selection_call.cpp"
// #include "mixture_call.cpp"
#include "fusion_call.cpp"

using namespace std;


int main () {	

	double fit [2] = {0.0, 0.0};
	int N = 192; // meg
	// int N = 156; // fmri
	fit[0] = 0.0; fit[1] = 0.0;
  	sferes_call(fit, N, "data_meg/S3/", 0.000265578, 0.545996, 0, 0.899626, 0, 0.00186337, 0, 0.00347151);
  	std::cout << fit[0] << " " << fit[1] << std::endl;  
   	return 0;
}