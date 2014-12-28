#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <math.h>

using namespace std;

void alignToMedian(double *daArray, int iSize) {    
    double* dpSorted = new double[iSize];
    for (int i = 0; i < iSize; ++i) dpSorted[i] = daArray[i];
    for (int i = iSize - 1; i > 0; --i) {
        for (int j = 0; j < i; ++j) {
            if (dpSorted[j] > dpSorted[j+1]) {
                double dTemp = dpSorted[j];
                dpSorted[j] = dpSorted[j+1];
                dpSorted[j+1] = dTemp;
            }
        }
    }
    double dMedian = dpSorted[(iSize/2)-1]+(dpSorted[iSize/2]-dpSorted[(iSize/2)-1])/2.0;    
    for (int i=0;i<iSize;i++) {daArray[i] = daArray[i]-dMedian;dpSorted[i] = dpSorted[i]-dMedian;}
    double dQ1 = dpSorted[(iSize/4)-1]+((dpSorted[(iSize/4)]-dpSorted[(iSize/4)-1])/2.0);
    double dQ3 = dpSorted[(iSize/4)*3-1]+((dpSorted[(iSize/4)*3+1]-dpSorted[(iSize/4)*3-1])/2.0);
    // std::cout << dpSorted[((iSize/4)*3)-2] << std::endl;
    // std::cout << dpSorted[((iSize/4)*3)-1] << std::endl;
    // // std::cout << dQ3 << std::endl;
    // std::cout << dpSorted[(iSize/4)*3] << std::endl;
    // std::cout << dpSorted[(iSize/4)*3+1] << std::endl;
    delete [] dpSorted;
    for (int i=0;i<iSize;i++) daArray[i] = daArray[i]/(dQ3-dQ1);    
}
void softmax(double *p, double *v, double b) {
	double sum = 0.0;
	double tmp[5];
	// for (int i=0;i<5;i++) std::cout << v[i] << " "; std::cout << std::endl;
	for (int i=0;i<5;i++) {
		tmp[i] = exp(v[i]*b);
		sum+=tmp[i];		
	}		
	for (int i=0;i<5;i++) {
		if (isinf(tmp[i])) {
			for (int j=0;j<5;j++) {
				p[j] = 0.0000001;
			}			
			p[i] = 0.9999996;
			return ;
		}	
	}	
	// for (int i=0;i<5;i++) std::cout << tmp[i] << " "; std::cout << std::endl;
	for (int i=0;i<5;i++) {
		p[i] = tmp[i]/sum;		
	}
}
double entropy(double *p) {
	double tmp = 0.0;
	for (int i=0;i<5;i++) {tmp+=p[i]*log2(p[i]);}
	return -tmp;
}
// void sferes_call(double * fit, const char* data_dir, double alpha_, double beta_)
void sferes_call(double * fit, const int N, const char* data_dir, double alpha_, double beta_, double sigma_)
{
	///////////////////	
	// parameters
	double alpha=0.0+alpha_*(1.0-0.0); //alpha +
	// double gamma=0.0+gamma_*(0.99-0.00); //alpha -
	double beta=0.0+beta_*(100.0-0.0);
	double sigma=0.0+(20.0-0.0)*sigma_;
	// double omega=0.0+omega_*(0.999999-0.0);
	
	int nb_trials = N/4;
	int n_state = 3;
	int n_action = 5;
	int n_r = 2;	
	///////////////////
	int sari [N][4];	
	double mean_rt [15];
	double mean_model [15];	
	double values [N]; // action probabilities according to subject
	double rt [N]; // rt du model	
	double p_a_mf [n_action];

	const char* _data_dir = data_dir;
	std::string file1 = _data_dir;
	std::string file2 = _data_dir;
	file1.append("sari.txt");
	file2.append("mean.txt");	
	std::ifstream data_file1(file1.c_str());
	string line;
	if (data_file1.is_open())
	{ 
		for (int i=0;i<N;i++) 
		{  
			getline (data_file1,line);			
			stringstream stream(line);
			std::vector<int> values(
     			(std::istream_iterator<int>(stream)),
     			(std::istream_iterator<int>()));
			for (int j=0;j<4;j++)
			{
				sari[i][j] = values[j];
			}
		}
	data_file1.close();	
	}
	std::ifstream data_file2(file2.c_str());	
	if (data_file2.is_open())
	{
		for (int i=0;i<15;i++) 
		{  
			getline (data_file2,line);			
			double f; istringstream(line) >> f;
			mean_rt[i] = f;
		}
	data_file2.close();	
	}		
	for (int i=0;i<4;i++)	
	{		
		// START BLOC //
		double values_mf [n_state][n_action];	
		int s, a, r;		
		double Hf = 0.0;
		for (int n=0;n<n_state;n++) { 			
			for (int m=0;m<n_action;m++) {
				values_mf[n][m] = 0.0;
			}
		}		
		// START TRIAL //
		for (int j=0;j<nb_trials;j++) 		
		{							
			// COMPUTE VALUE
			s = sari[j+i*nb_trials][0]-1;
			a = sari[j+i*nb_trials][1]-1;
			r = sari[j+i*nb_trials][2];							
			softmax(p_a_mf, values_mf[s], beta);
			double Hf = entropy(p_a_mf);

			// int ind=-1;
			// for (int n=0;n<5;n++) {
			// 	if (isnan(p_a_mf[n])) {
			// 		ind = n;
			// 		break;
			// 	}
			// }
			// if (ind!=-1) {
			// 	for (int n=0;n<5;n++) {
			// 		p_a_mf[n] = 0.0001;
			// 	}
			// }
			// p_a_mf[ind] = 0.9996;						
			// if (isnan(Hf)) Hf = 0.005;
			
			values[j+i*nb_trials] = log(p_a_mf[a]);						
			rt[j+i*nb_trials] =  Hf;
			// MODEL FREE	
			double reward;
			if (r == 0) {reward = -1.0;} else {reward = 1.0;}
			double delta = reward - values_mf[s][a];
			values_mf[s][a]+=(alpha*delta);
			// if (r==1)				
			// 	values_mf[s][a]+=(alpha*delta);
			// else if (r==0)
			// 	values_mf[s][a]+=(omega*delta);
		}
	}
	
	// ALIGN TO MEDIAN
	alignToMedian(rt, N);	
	// for (int i=0;i<N;i++) std::cout << rt[i] << std::endl;
	double tmp2[15];
	for (int i=0;i<15;i++) {
		mean_model[i] = 0.0;
		tmp2[i] = 0.0;
	}

	for (int i=0;i<N;i++) {
		mean_model[sari[i][3]-1]+=rt[i];
		tmp2[sari[i][3]-1]+=1.0;				
	}	
	double error = 0.0;
	for (int i=0;i<15;i++) {
		mean_model[i]/=tmp2[i];
		error+=pow(mean_rt[i]-mean_model[i],2.0);		
	}	
	for (int i=0;i<N;i++) fit[0]+=values[i];
	fit[1] = -error;
	
	if (isnan(fit[0]) || isinf(fit[0]) || isinf(fit[1]) || isnan(fit[1]) || fit[0]<-10000 || fit[1]<-10000) {
		fit[0]=-1000.0;
		fit[1]=-1000.0;
		return;
	}
	else {
		fit[0]+=2000.0;
		fit[1]+=500.0;
		return ;
	}
}