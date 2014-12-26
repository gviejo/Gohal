#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <math.h>
#include <boost/math/distributions.hpp>

using namespace std;

void choldc1(int n, double (*a) [15], double *p) {
	int i,j,k;
	double sum;
	for (i = 0; i < n; i++) {
		for (j = i; j < n; j++) {
			sum = a[i][j];
			for (k = i - 1; k >= 0; k--) {
				sum -= a[i][k] * a[j][k];
			}
			if (i == j) {
				if (sum <= 0) {
					printf(" a is not positive definite!\n");
				}
				p[i] = sqrt(sum);
			}
			else {
				a[j][i] = sum / p[i];
			}
		}
	}
}
void choldc(int n, double (*A) [15], double (*a) [15]) {
	int i,j;
	double p[15];
    for (i = 0; i < n; i++) 
	    for (j = 0; j < n; j++) 
	      	a[i][j] = A[i][j];	
	choldc1(n, a, p);
    for (i = 0; i < n; i++) {
        a[i][i] = p[i];
        for (j = i + 1; j < n; j++) {
            a[i][j] = 0;
	    }
	}
}
double cdfcall(double value, double mean, double var)
{
	boost::math::normal_distribution<> myNormal(mean, var);
   	return cdf(myNormal, value);
}
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

    delete [] dpSorted;
    for (int i=0;i<iSize;i++) daArray[i] = daArray[i]/(dQ3-dQ1);    
}
void softmax(double *p, double *v, double b) {
	double sum = 0.0;
	double tmp[5];
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
	for (int i=0;i<5;i++) {
		p[i] = tmp[i]/sum;		
	}
}
double entropy(double *p) {
	double tmp = 0.0;
	for (int i=0;i<5;i++) {tmp+=p[i]*log2(p[i]);}
	return -tmp;
}
void computevpi(double *vpi, double *mean, double *cov) {
	double sorted[5];
	int ind[5];
	for (int i=0;i<5;i++) {sorted[i] = mean[i]; ind[i] = i;}	
	for (int i=5-1;i>0;--i) {
		for (int j=0;j<i;++j) {
			if (sorted[j] > sorted[j+1]) {
				double tmp = sorted[j];
				int tmp2 = ind[j];
				sorted[j] = sorted[j+1];
				sorted[j+1] = tmp;
				ind[j] = ind[j+1];
				ind[j+1] = tmp2;
			}
		}
	}
	
	vpi[ind[4]] = (mean[ind[3]]-mean[ind[4]])*cdfcall(mean[ind[3]], mean[ind[4]], sqrt(cov[ind[4]])) +
				  (sqrt(cov[ind[4]])/sqrt(2.0*M_PI))*exp(-pow((mean[ind[3]]-mean[ind[4]]),2)/(2*cov[ind[4]]));

	for (int i=3;i>-1;i--) {		
		vpi[ind[i]] = (mean[ind[i]]-mean[ind[4]])*(1.0-cdfcall(mean[ind[4]], mean[ind[i]], sqrt(cov[ind[i]]))) +
				  (sqrt(cov[ind[i]])/sqrt(2.0*M_PI))*exp(-pow((mean[ind[4]]-mean[ind[i]]),2)/(2*cov[ind[i]]));
		
	}
	
}
void sigmapoint(double (*point) [15], double *weight, double (*values) [5], double (*cov) [15], double kappa) {
	const int  n = 15;
	double tmp[n][n];
	for (int i=n+1;i<2*n+1;i++) {
		for (int j=0;j<15;j++) {
			point[i][j] = 0.0;
		}		
	}
	double c[n][n];
	for (int i=0;i<3;i++) {
		for (int j=0;j<5;j++) {
			point[0][i*5+j] = values[i][j];
		}
	}
	for (int i=0;i<n;i++) {
		for (int j=0;j<n;j++) {
			tmp[i][j] = (n+kappa)*cov[i][j];	
		}
	}
	choldc(n, tmp, c);

	for (int i=1;i<16;i++) {
		for (int j=0;j<n;j++) {
			point[i][j] = values[j/5][j%5]+c[j][i-1];
		}
	}
	for (int i=16;i<31;i++) {
		for (int j=0;j<n;j++) {
			point[i][j] = values[j/5][j%5]-c[j][i-16];
		}
	}

	weight[0] = 0.0;
	for (int i=1;i<2*n+1;i++) {
		weight[i] = 1./(2*n+kappa);
	}
}
// void sferes_call(double * fit, const char* data_dir, double beta_, double eta_, double length_, double threshold_, double noise_, double sigma_, doubl)
void sferes_call(double * fit, const char* data_dir, double beta_, double eta_, double length_, double threshold_, double noise_, double sigma_, double sigma_rt_)
{
	///////////////////
	double max_entropy = -log2(0.2);
	// parameters
	double beta=0.0+beta_*(100.0-0.0);
	double eta=0.00001+eta_*(0.001-0.00001);
	double noise=0.0+noise_*(0.1-0.0);
	int length=1+(10-1)*length_;	
	double threshold=0.01+(max_entropy-0.01)*threshold_;
	double sigma=0.0+(1.0-0.0)*sigma_;	
	double sigma_rt=0.0+(10.0-0.0)*sigma_rt_;	
	double init_cov = 10.0;
	double kappa = 0.1;
	double var_obs = 0.05;
	const int n_state = 3;
	const int n_action = 5;
	const int n_r = 2;	

	///////////////////
	int sari [156][4];	
	double mean_rt [15];
	double mean_model [15];	
	double values [156]; // action probabilities according to subject
	double rt [156]; // rt du model	
	double p_a [n_action];

	const char* _data_dir = data_dir;
	std::string file1 = _data_dir;
	std::string file2 = _data_dir;
	file1.append("sari.txt");
	file2.append("mean.txt");	
	std::ifstream data_file1(file1.c_str());
	string line;
	if (data_file1.is_open())
	{ 
		for (int i=0;i<156;i++) 
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
		double p_s [length][n_state];
		double p_a_s [length][n_state][n_action];
		double p_r_as [length][n_state][n_action][n_r];				
		double p [n_state][n_action][2];		
		double values_mf [n_state][n_action];	
		double values_mb [n_action];
		double tmp [n_state][n_action][2];
		double p_ra_s [n_action][2];
		double p_a_rs [n_action][2];
		double p_r_s [2];
		double vpi[n_action];
		double diagonal[n_action];
		double point [2*n_state*n_action+1][n_state*n_action];
		double weight[2*n_state*n_action+1];
		double rewards_predicted[2*n_state*n_action+1];
		double reward_predicted = 0.0;
		double cov_values_rewards[15];
		double cov_rewards;
		double kalman_gain[15];
		double reward_rate[n_state];
		int n_element = 0;
		int s, a, r;
		double Hf, Hb = max_entropy;
		for (int n=0;n<n_state;n++) { 			
			reward_rate[n] = 0.0;
			for (int m=0;m<n_action;m++) {
				values_mf[n][m] = 0.0;
			}
		}		
		// CREATE COVARIANCE 
		double cov [15][15];
		double noise_cov [15][15];
		for (int n=0;n<15;n++) {
			for (int m=0;m<15;m++) {
				cov[n][m] = 0.0;
				noise_cov[n][m] = 0.0;
				if (n==m) {cov[n][m] = 1.0; noise_cov[n][m] = init_cov*eta;}
			}
			
		}
		// START TRIAL //		
		for (int j=0;j<39;j++) 				
		{				
			
			s = sari[j+i*39][0]-1;
			a = sari[j+i*39][1]-1;
			r = sari[j+i*39][2];										
			//PREDICTION STEP
			for (int n=0;n<15;n++) {
				for (int m=0;m<15;m++) {
					noise_cov[n][m]=cov[n][m]*eta;
					cov[n][m] += noise_cov[n][m];
				}
			}
			int nb_inferences = 0;									
			softmax(p_a, values_mf[s], beta);
			for (int n=0;n<n_action;n++) diagonal[n] = cov[n_action*s+n][n_action*s+n];
			computevpi(vpi, values_mf[s], diagonal);

			int k = 0;

			for (int u=0;u<5;u++) {
				if (vpi[u] > reward_rate[s]) {
					// UNIFORM					
					for (int n=0;n<n_state;n++){
						for (int m=0;m<n_action;m++) {
							p[n][m][0] = 1./30; p[n][m][1] = 1./30; 
						}
					}
					for (int n=0;n<n_action;n++){						
						values_mb[n] = 1./n_action;								
					}						
					Hb = max_entropy;					
					while ( Hb > threshold && nb_inferences < n_element) {						
						// INFERENCE				
						double sum = 0.0;
						for (int n=0;n<3;n++) {
							for (int m=0;m<5;m++) {
								for (int o=0;o<2;o++) {
									p[n][m][o] += (p_s[k][n] * p_a_s[k][n][m] * p_r_as[k][n][m][o]);
									sum+=p[n][m][o];
								}
							}
						}
						for (int n=0;n<3;n++) {
							for (int m=0;m<5;m++) {
								for (int o=0;o<2;o++) {
									tmp[n][m][o] = (p[n][m][o]/sum);
								}
							}
						}
						nb_inferences+=1;
						// EVALUATION
						sum = 0.0;				
						for (int m=0;m<5;m++) {
							for (int o=0;o<2;o++) {
								p_r_s[o] = 0.0;
								sum+=tmp[s][m][o];						
							}
						}
						for (int m=0;m<5;m++) {
							for (int o=0;o<2;o++) {
								p_ra_s[m][o] = tmp[s][m][o]/sum;
								p_r_s[o]+=p_ra_s[m][o];						
							}
						}
						sum = 0.0;
						for (int m=0;m<5;m++) {
							for (int o=0;o<2;o++) {
								p_a_rs[m][o] = p_ra_s[m][o]/p_r_s[o];
							}
							values_mb[m] = p_a_rs[m][1]/p_a_rs[m][0];
							sum+=values_mb[m];
						}																				
						for(int m=0;m<5;m++) {
							p_a[m] = values_mb[m]/sum;
						}				
						Hb = entropy(p_a);
						k+=1;						
					}	
					break;
				}
				
			}
			double H = entropy(p_a);
			double N = nb_inferences+1;


			// int ind=-1;
			// for (int n=0;n<5;n++) {
			// 	if (isnan(p_a[n])) {
			// 		ind = n;
			// 		break;
			// 	}
			// }
			// if (ind!=-1) {
			// 	for (int n=0;n<5;n++) {
			// 		p_a[n] = 0.0001;
			// 	}
			// 	p_a[ind] = 0.9996;
			// }
			

			// if (isnan(H)) H = 0.005;
			
			values[j+i*39] = log(p_a[a]);

			// rt[j+i*39] = H*sigma_rt+log2(N);
			rt[j+i*39] =  pow(N, sigma_rt)+H;

			// UPDATE MEMORY 						
			for (int k=length-1;k>0;k--) {
				for (int n=0;n<3;n++) {
					p_s[k][n] = p_s[k-1][n]*(1.0-noise)+noise*(1.0/n_state);
					for (int m=0;m<5;m++) {
						p_a_s[k][n][m] = p_a_s[k-1][n][m]*(1.0-noise)+noise*(1.0/n_action);
						for (int o=0;o<2;o++) {
							p_r_as[k][n][m][o] = p_r_as[k-1][n][m][o]*(1.0-noise)+noise*0.5;				
						}
					}
				}
			}						
			if (n_element < length) n_element+=1;
			for (int n=0;n<3;n++) {
				p_s[0][n] = 0.0;
				for (int m=0;m<5;m++) {
					p_a_s[0][n][m] = 1./n_action;
					for (int o=0;o<2;o++) {
						p_r_as[0][n][m][o] = 0.5;
					}
				}
			}			
			p_s[0][s] = 1.0;
			for (int m=0;m<5;m++) {
				p_a_s[0][s][m] = 0.0;
			}
			p_a_s[0][s][a] = 1.0;
			p_r_as[0][s][a][(r-1)*(r-1)] = 0.0;
			p_r_as[0][s][a][r] = 1.0;
			// MODEL FREE	
			double reward;
			if (r == 0) {reward = -1.0;} else {reward = 1.0;}
			//COMPUTE SIGMA POINTS			
			sigmapoint(point, weight, values_mf, cov, kappa);


			for (int n=0;n<2*n_state*n_action+1;n++) rewards_predicted[n] = point[n][n_action*s+a];		

			// DOT PRODUCT rewards_predicted*weights
			reward_predicted = 0.0;
			for (int n=0;n<2*n_state*n_action+1;n++) reward_predicted+=(weight[n]*rewards_predicted[n]);						
			
			for (int m=0;m<15;m++) cov_values_rewards[m] = 0.0;							

			double tmp3 [2*n_state*n_action+1][15];
			for (int n=0;n<2*n_state*n_action+1;n++) {
				for (int m=0;m<15;m++) {
					tmp3[n][m] = weight[n]*(point[n][m]-values_mf[m/5][m%5])*(rewards_predicted[n]-reward_predicted);
				}			
			}

			for (int m = 0; m<15;m++) {
				for (int n=0;n<2*n_state*n_action+1;n++) {
					cov_values_rewards[m]+=tmp3[n][m];
				}
			}
			cov_rewards = 0 ;
			for (int n=0;n<2*n_state*n_action+1;n++) cov_rewards += weight[n]*pow((rewards_predicted[n]-reward_predicted),2);
			cov_rewards+=var_obs;

			for (int n=0;n<15;n++) {
				kalman_gain[n] = cov_values_rewards[n]/cov_rewards;
			}			
			//UPDATING			
			for (int n=0;n<3;n++) {
				for (int m=0;m<5;m++) {
					values_mf[n][m] += kalman_gain[m+n*5]*(reward-reward_predicted);
					for (int o=0;o<15;o++) {
						cov[m+n*5][o] -= (kalman_gain[m+n*5]*cov_rewards*kalman_gain[o]);
					}					
				}
			}
			// REWARD RATE
			reward_rate[s] = (1.0-sigma)*reward_rate[s]+sigma*reward;
		}
	}	
	// ALIGN TO MEDIAN
	alignToMedian(rt, 156);	

	double tmp2[15];
	for (int i=0;i<15;i++) {
		mean_model[i] = 0.0;
		tmp2[i] = 0.0;
	}

	for (int i=0;i<156;i++) {
		mean_model[sari[i][3]-1]+=rt[i];
		tmp2[sari[i][3]-1]+=1.0;				
	}	
	double error = 0.0;
	for (int i=0;i<15;i++) {
		mean_model[i]/=tmp2[i];
		error+=pow(mean_rt[i]-mean_model[i],2.0);		
	}	
	for (int i=0;i<156;i++) fit[0]+=values[i];	
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