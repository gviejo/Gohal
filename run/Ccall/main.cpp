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
	// int N = 192; // meg
	int N = 156; // fmri
	fit[0] = 0.0; fit[1] = 0.0;
  	// sferes_call(fit, N, "data_meg/S10/", 0.91922, 0.00637459, 0.172274, 0.559175, 1, 0.341147, 0.00225426, 0.249983);
  	sferes_call(fit, N, "data_fmri/S11/", 0.261945, 0.0227431, 1, 0.58308, 0.0176517, 0.149967, 0.00904301, 0.809806);
  	std::cout << fit[0] << " " << fit[1] << std::endl;  
   	return 0;
}