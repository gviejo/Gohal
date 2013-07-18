import numpy as np
from scipy.stats import norm
from prettytable import PrettyTable
import operator
from copy import deepcopy
import cPickle as pickle
import scipy.io
from scipy import misc
from scipy import stats
from scipy.stats import binom, sem

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

def print_dict(dictionary, ident = '', braces=1):
    """ Recursively prints nested dictionaries."""
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            print '%s%s%s%s' %(ident,braces*'[',key,braces*']') 
            print_dict(value, ident+'  ', braces+1)
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

def computeEntropy(values, beta):
    tmp = np.exp(values*float(beta))
    tmp = tmp/float(np.sum(tmp))
    return -np.sum(tmp*np.log2(tmp))

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
                
def getRepresentativeSteps(data, stimulus, action, responses):
    m, n = data.shape
    bad_trials = 0
    incorrect_trials = 0
    assert(data.shape == stimulus.shape == responses.shape)

    indice = np.zeros((m,n))

    for i in xrange(m):
        incorrect = dict()

        for j in [1,2,3]:
            if len(np.where((responses[i] == 1) & (stimulus[i] == j))[0]):
                correct = np.where((responses[i] == 1) & (stimulus[i] == j))[0][0]
                t = len(np.unique(action[i][np.where((responses[i][0:correct] == 0) & (stimulus[i][0:correct] == j))[0]]))
                incorrect[t] = j
            else:
                bad_trials+=1
                continue
            
        if 1 in incorrect.keys() and 3 in incorrect.keys() and 4 in incorrect.keys():            
            first = incorrect[1]
            second = incorrect[3]
            third = incorrect[4]
        else:
            incorrect_trials+=1
            continue
            
        #extracting first wrongs
        first_action = []
        for j in [first, second, third]:
            first_action.append(action[i, np.where(stimulus[i] == j)[0][0]])
            indice[i, np.where((responses[i] == 0) & (stimulus[i] == j) & (action[i] == first_action[-1]))[0]] = 1

        #second step+first right+persistance for [second, third]
        indice[i,np.where(stimulus[i] == first)[0][1]] = 2
        second_action = []
        for j in [second, third]:
            second_action.append(action[i, np.where((stimulus[i] == j) & (indice[i] == 0))[0][0]])
            indice[i,np.where((stimulus[i] == j) & (action[i] == second_action[-1]) & (responses[i] == 0) & (indice[i] == 0))] = 2

        #third step
        third_action = []
        for j,k in zip([second, third], second_action):
            third_action.append(action[i,np.where((stimulus[i] == j) & (action[i] <> k) & (indice[i] == 0))[0][0]])
            indice[i, np.where((stimulus[i] == j) & (action[i] == third_action[-1]) & (indice[i] == 0))[0]] = 3
                
        #fourth step + second right
        indice[i, np.where((stimulus[i] == second) & (responses[i] == 1) & (indice[i] == 0))[0][0]] = 4
        if len(np.where((stimulus[i] == third) & (action[i] <> third_action[1]) & (indice[i] == 0) & (responses[i] == 0))[0]) > 0:
            fourth_action = action[i, np.where((stimulus[i] == third) & (action[i] <> third_action[1]) & (indice[i] == 0))[0][0]]
            indice[i, np.where((stimulus[i] == third) & (action[i] == fourth_action) & (indice[i] == 0))[0]] = 4

        #fifth step
        indice[i, np.where((stimulus[i] == third) & (responses[i] == 1) & (indice[i] == 0))[0][0]] = 5

        #indicing for correct first from 6 to ...
        for k in [first, second, third]:
            tmp = 6
            for j in np.where((stimulus[i] == k) & (responses[i] == 1) & (indice[i] == 0))[0]:
                indice[i,j] = tmp; tmp += 1

        #setting maintenance errors
        for k in [first, second, third]:
            essai_correct = np.where((stimulus[i] == k) & (responses[i] == 1))[0][0]
            for j in xrange(essai_correct, len(stimulus[i])):
                if stimulus[i, j] == k and responses[i,j] == 0:
                    indice[i,j] = 0

    steps = dict()
    for i in range(1,16):
        steps[i] = data[indice == i]
    if incorrect_trials:
        print "Number of incorrect trials : %.2f" % (incorrect_trials/float(m)) + " %"
    if bad_trials:
        print "Number of bad trials : %.2f" % (bad_trials/float(m)) + " %"

    return steps, indice

 

def computeMeanRepresentativeSteps(data):
    assert(type(data) == dict)
    m = []
    s = []
    for i in data.iterkeys():
        if len(data[i]):
            m.append(np.mean(data[i]))
            s.append(sem(data[i]))
                    
    return np.array(m), np.array(s)
            
    
def extractStimulusPresentation(data, stimulus, action, responses):
    tmp = dict({1:[],2:[],3:[]})
    m, n = responses.shape
    assert(stimulus.shape == responses.shape == action.shape == data.shape)
    bad_trials = 0
    incorrect_trials = 0

    for i in xrange(m):
        incorrect = dict()

        for j in [1,2,3]:
            if len(np.where((responses[i] == 1) & (stimulus[i] == j))[0]):
                correct = np.where((responses[i] == 1) & (stimulus[i] == j))[0][0]
                t = len(np.unique(action[i][np.where((responses[i][0:correct] == 0) & (stimulus[i][0:correct] == j))[0]]))
                incorrect[t] = j
            else:
                bad_trials += 1
                continue
            
        if 1 in incorrect.keys() and 3 in incorrect.keys() and 4 in incorrect.keys():            
            first = incorrect[1]
            second = incorrect[3]
            third = incorrect[4]
            tmp[1].append(data[i,stimulus[i] == first][0:10])
            tmp[2].append(data[i,stimulus[i] == second][0:10])
            tmp[3].append(data[i,stimulus[i] == third][0:10])

        else:
            incorrect_trials += 1
            continue

    final = dict({'mean':[],'sem':[]})

    for i in tmp.keys():
        tmp[i] = np.array(tmp[i])
        final['mean'].append(np.mean(tmp[i], 0))
        final['sem'].append(sem(tmp[i], 0))
    final['mean'] = np.array(final['mean'])
    final['sem'] = np.array(final['sem'])

    if incorrect_trials:
        print "Number of incorrect trials : %.2f" % (incorrect_trials/float(m))+" %"
    if bad_trials:
        print "Number of bad trials : %.2f" % (bad_trials/float(m))+" %"

    return final

def extractStimulusPresentation2(data, stimulus, action, responses):
    tmp = dict({1:[],2:[],3:[]})
    m, n = responses.shape
    assert(stimulus.shape == responses.shape == action.shape == data.shape)
    bad_trials = 0
    incorrect_trials = 0
    for i in xrange(m):
        incorrect = dict()
        for j in [1,2,3]:
            if len(np.where((responses[i] == 1) & (stimulus[i] == j))[0]):
                correct = np.where((responses[i] == 1) & (stimulus[i] == j))[0][0]
                t = len(np.unique(action[i][np.where((responses[i][0:correct] == 0) & (stimulus[i][0:correct] == j))[0]]))
                incorrect[t] = j
            else:
                bad_trials += 1
                continue            
        if 1 in incorrect.keys() and 3 in incorrect.keys() and 4 in incorrect.keys():            
            first = incorrect[1]
            second = incorrect[3]
            third = incorrect[4]
            tmp[1].append(data[i,stimulus[i] == first][0:10])
            tmp[2].append(data[i,stimulus[i] == second][0:10])
            tmp[3].append(data[i,stimulus[i] == third][0:10])
        else:
            incorrect_trials += 1
            continue
    for i in tmp.keys():
        tmp[i] = np.array(tmp[i])
    return tmp

    
def convertStimulus(state):
    return (state == 's1')*1+(state == 's2')*2 + (state == 's3')*3
def convertAction(action):
    return (action=='thumb')*1+(action=='fore')*2+(action=='midd')*3+(action=='ring')*4+(action=='little')*5
