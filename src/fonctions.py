import numpy as np
from scipy.stats import norm
from matplotlib import *
from prettytable import PrettyTable
import operator
from copy import deepcopy
import cPickle as pickle
import scipy.io
from scipy import misc
from scipy import stats

def displayQValues(states, actions, values, ind = 0):
    foo = PrettyTable()
    line_actions = ["State\Action"]
    for i in actions:
        line_actions.append(i)
    foo.set_field_names(line_actions)
    for j in states:
        line = [j]
        for v in actions:
            line.append(round(values[ind][values[(j,v)]], 3))
        foo.add_row(line)
    foo.printt()

def print_dict(self, dictionary, ident = '', braces=1):
    """ Recursively prints nested dictionaries."""
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            print '%s%s%s%s' %(ident,braces*'[',key,braces*']') 
            self.print_dict(value, ident+'  ', braces+1)
        else:
            print ident+'%s = %s' %(key, value)


def createQValuesDict(states, actions):
    #return a big dict
    values = {0:np.zeros(len(states)*len(actions))}
    tmp = 0
    for s in states:
        values[s] = []
        n = 0
        for a in actions:
            values[(s,a)] = tmp
            values[s].append(tmp)
            values[(s,n)] = a
            tmp = tmp + 1
            n = n + 1
    return values

def createRewardRateDict():
    return dict({'rate':0,
                 'reward':0,
                 'tau':0,
                 'step':0,
                 'tau_list':[0],
                 'r_list':[0],
                 ('s0', 'pl'):8,
                 ('s0', 'em'):12,
                 ('s1', 'pl'):12,
                 ('s1', 'em'):0,
                 'r*tau':[]})

def createCovarianceDict(n, init_cov, eta):
    #cov = np.eye(n)*init_cov+(1-np.eye(n,n))*((1-init_cov)/(n-1))    
    cov = np.eye(n)
    return dict({'cov':cov,
                 'noise':np.eye(n,n)*init_cov*eta})


def getBestAction(state, values, ind = 0):
    return values[(state, np.where(values[ind][values[state]] == np.max(values[ind][values[state]]))[0][0])]

def getBestActionSoftMax(state, values, beta, ind = 0):
    tmp = np.exp(values[ind][values[state]]*float(beta))
    tmp = tmp/float(np.sum(tmp))
    tmp = [np.sum(tmp[0:i]) for i in range(len(tmp))]
    return values[(state, np.sum(np.array(tmp) < np.random.rand())-1)]

def computeVPIValues(mean, variance):
    #WARNING input and output very specific
    # mean = array(current state), variance = array(current state)
    #vpi = array(current state)
    vpi = np.zeros((len(mean)))
    ind = np.argsort(mean)
    vpi[ind[-1]] = (mean[ind[-2]]-mean[ind[-1]])*norm.cdf(mean[ind[-2]], mean[ind[-1]], np.sqrt(variance[ind[-1]])) + (np.sqrt(variance[ind[-1]])/np.sqrt(2*np.pi))*np.exp(-(mean[ind[-2]]-mean[ind[-1]])**2/(2*variance[ind[-1]]))
    for i in range(len(mean)-2, -1, -1):
        vpi[ind[i]] = (mean[ind[i]]-mean[ind[-1]])*(1-norm.cdf(mean[ind[-1]], mean[ind[i]], np.sqrt(variance[ind[i]]))) + (np.sqrt(variance[ind[i]])/np.sqrt(2*np.pi))*np.exp(-(mean[ind[-1]]-mean[ind[i]])**2/(2*variance[ind[i]]))        
    return vpi
       
def updateRewardRate(reward_rate, sigma, delay = 0.0):
    return ((1-sigma)**(1+delay))*reward_rate['rate']+sigma*reward_rate['reward']

def updateQValuesHabitual(values, delta, alpha):
    return values+delta*alpha

def computeSigmaPoints(values, covariance, kappa=0.5):
    n = len(values)
    point = np.zeros((2*n+1,n))
    point[0] = values
    c = np.linalg.cholesky((n+kappa)*covariance)
    point[range(1,n+1)] = values+np.transpose(c)
    point[range(n+1, 2*n+1)] = values-np.transpose(c)
    weights = np.zeros((2*n+1,1))
    weights[0] = kappa/(n+kappa)
    weights[1:2*n+1] = 1/(2*n+kappa)
    return point, weights

def computeExplorationCost(reward_rate, tau, transition):
    return tau*reward_rate[transition]
    
def createTransitionDict(state1, action, state2, stopstate):
    transition = dict()
    n = float(len(set(state1)))
    transition[None] = stopstate
    for i,j,k in zip(state1, action, state2):
        transition[(i,j,k)] = 1/n
        transition[(i,j)] = k
    return transition

def updateTransition(transition_values, transition, phi):
    transition_values[transition] = (1-phi)*transition_values[transition] + phi
    for i in transition_values.iterkeys():
        if i <> transition and i <> None and len(i) == 3 and i[0] == transition[0]:
            transition_values[i] = (1-phi)*transition_values[i]
    return transition_values

def updateRewardsFunction(rewards_function, state, action, rau):
    rewards_function[1] = (1-rau)*rewards_function[1]+rau*rewards_function[0][rewards_function[(state, action)]]
    return rewards_function

def computeGoalValue(values, state, action, rewards, gamma, depth, phi, rau):
    rewards_function = deepcopy(rewards)
    rewards_function[1] = rewards_function[0].copy()
    rewards_function = updateRewardsFunction(rewards_function, state, action, rau)
    transition = createTransitionDict(['s0','s0','s1','s1'],['pl','em','pl','em'],['s1','s0','s0',None], 's0') #<====VERY BAD==============    NEXT_STATE = TRANSITION[(STATE, ACTION)]
    next_state = transition[(state, action)]
    if next_state == None:
        return rewards_function[1][rewards_function[(state, action)]] + gamma*transition[(state, action, next_state)]*np.max(values[0][values[transition[None]]])        
    else:
        transition = updateTransition(transition, (state, action, next_state), phi)
        tmp = np.max([computeGoalValueRecursive(values, next_state, a, rewards_function.copy(), transition.copy(), gamma, depth-1, phi, rau) for a in values[next_state]])
        value = rewards_function[1][rewards_function[(state, action)]] + gamma*transition[(state, action, next_state)]*tmp
        return value

def computeGoalValueRecursive(values, state, a, rewards_function, transition, gamma, depth, phi, rau):
    action = values[(state, values[state].index(a))]
    next_state = transition[(state, action)]
    rewards_function = updateRewardsFunction(rewards_function, state, action, rau)
    transition = updateTransition(transition, (state, action, next_state), phi)
    if next_state == None:
        return rewards_function[1][rewards_function[(state, action)]] + gamma*transition[(state, action, next_state)]*np.max(values[0][values[transition[None]]])        
    elif depth == 0:
        return rewards_function[1][rewards_function[(state, action)]] + gamma*transition[(state, action, next_state)]*np.max(values[0][values[next_state]])
    else:
        tmp = np.max([computeGoalValueRecursive(values, next_state, a, rewards_function.copy(), transition.copy(), gamma, depth-1, phi, rau) for a in values[next_state]])
        return rewards_function[1][rewards_function[(state, action)]] + gamma*transition[(state, action, next_state)]*tmp


def testQValues(states, values, beta, ind, niter):
    tmp = np.zeros((4)) #VBAAAAAAD
    for i in range(niter):
        for s in states:
            a = getBestActionSoftMax(s, values, beta, ind)
            tmp[values[(s,a)]] = tmp[values[(s, a)]] + 1 
    tmp = tmp/float(niter)
    return tmp
                

def saveData(file_name, data):
    f = open(file_name, 'w')
    pickle.dump(data, f, protocol = 2)
    f.close()

def loadData(file_name):
    f = open(file_name, 'r')
    data = pickle.load(f)
    f.close()
    return data

def loadDirectoryMEG(direct):
    data = dict()
    line = "ls "+direct
    p = os.popen(line, "r").read()
    files = p.split('\n')[:-1]    
    for i in files:
        tmp = scipy.io.loadmat(direct+i+'/beh.mat')['beh']
        data[i] = dict()
        for j in range(1, len(tmp[0])-1):
            data[i][j] = {}
            for k in range(len(tmp.dtype.names)):
                data[i][j][tmp.dtype.names[k]] = tmp[0][j][k]   
    return data

def loadDirectoryfMRI(direct):
    data = dict()
    tmp = scipy.io.loadmat(direct+'/beh_allSubj.mat')['data']
    m, n = tmp.shape
    for i in xrange(m):        
        data['S'+str(i+1)] = dict()
        for j in xrange(n):
            data['S'+str(i+1)][j+1] = dict()
            for k in range(len(tmp[i][j].dtype.names)):
                if tmp[i][j].dtype.names[k] == 'sar_time':
                    data['S'+str(i+1)][j+1]['time'] = tmp[i][j][k]
                else:
                    data['S'+str(i+1)][j+1][tmp[i][j].dtype.names[k]] = tmp[i][j][k]                                   
    return data
        
def computeMeanReactionTime(data, case = None, ind = 40):
    tmp = []
    if case.lower() == 'fmri':
        for s in data.iterkeys():
            for b in data[s].iterkeys():
                tmp.append(data[s][b]['RT'].flatten()[0:ind])
    #TODO for meg
    reaction = dict({'mean':np.mean(tmp, 0),
                     'sem':stats.sem(tmp, 0)})
    return reaction
    
def getRepresentativeSteps(data, stimulus, responses):
    m, n = data.shape
    assert(data.shape == stimulus.shape == responses.shape)

    steps = dict()
    for s in xrange(n/3):
        steps[s] = []


    steps[1] = data[:,0:3][responses[:,0:3] == 0]


    for i in xrange(m):
        for j in xrange(n):
            if j < 3:
                steps[1].append(reaction)
       


    for i in xrange(1,4):
        tmp = []
        tmpr = []
        for j in xrange(m):
            tmp.append(data[j][stimulus[j] == i].copy())
            tmpr.append(responses[j][stimulus[j] == i].copy())
 
    steps = dict()
    for i in xrange(len(responses)):
        if np.where(responses[i] == 1)[0][0] in steps.keys():
            steps[np.where(responses[i] == 1)[0][0]].append(i)
        else:
            steps[np.where(responses[i] == 1)[0][0]] = [i]


    for i in xrange(m):
        ind = np.where(responses[i] == 1)[0]
        rest = set([1,2,3])
        #STEP 1
        steps[1].append(data[i][0:ind[0]].copy())
        first = stimulus[i][ind[0]]
        rest.remove(first)
        #STEP 2 first right
        steps[2].append(reaction[i][ind[0]].copy())
        steps[2].append(reaction[i][np.where(stimulus[i] == list(rest)[0])[0][1]].copy())
        steps[2].append(reaction[i][np.where(stimulus[i] == list(rest)[1])[0][1]].copy())
        #STEP 3
        


        

 
            
                

            
            
    


