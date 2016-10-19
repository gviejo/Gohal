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
  	sferes_call(fit, N, "data_meg/S10/", 1, 0.00885037, 1, 1, 0.0479494, 0.32029, 0.00357043, 1);
  	// sferes_call(fit, N, "data_fmri/S11/", 0.261945, 0.0227431, 1, 0.58308, 0.0176517, 0.149967, 0.00904301, 2.0);
  	// sferes_call(fit, N, "data_fmri/S9/", 0.261945, 0.0227431, 1, 1.0);
  	std::cout << fit[0] << " " << fit[1] << std::endl;  
   	return 0;
}