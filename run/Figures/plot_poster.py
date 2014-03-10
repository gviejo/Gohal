import sys
from optparse import OptionParser
import numpy as np
from matplotlib import rc_file
rc_file("figures.rc")
import matplotlib.pyplot as plt
sys.path.append("../../src")
from fonctions import *
from ColorAssociationTasks import CATS

from HumanLearning import HLearning
from matplotlib import *
from pylab import *
from Selection import *
from Models import *


# -----------------------------------
# HUMAN LEARNING
# -----------------------------------
human = HLearning(dict({'meg':('../../PEPS_GoHaL/Beh_Model/',42), 'fmri':('../../fMRI',39)}))
# -----------------------------------
def testModel(model):
    model.startExp()
    for i in xrange(nb_blocs):
        cats.reinitialize()
        model.startBloc()
        for j in xrange(nb_trials):
            sys.stdout.write("\r Bloc : %s | Trial : %i" % (i,j)); sys.stdout.flush()                    
            state = cats.getStimulus(j)
            action = model.chooseAction(state)
            reward = cats.getOutcome(state, action)
            model.updateValue(reward)

    model.state = convertStimulus(np.array(model.state))
    model.action = convertAction(np.array(model.action))
    model.responses = np.array(model.responses)
    model.reaction = np.array(model.reaction)

p_selection = {'beta': 4,
                'eta': 0.00020800000000000001,
                'gamma': 1.0,
                'length': 8.53298,
                'noise': 0.00321344,
                'sigma': 0.2,
                'sigma_bwm': 0.3,
                'sigma_ql': 0.060008,
                'threshold': 0.40118970201892307}
p_fusion = dict({'noise':0.0001,
                'length':9,
                'alpha':0.7,
                'beta':2.0,
                'gamma':0.4,
                'gain':1.0,
                'sigma_bwm':0.1,
                'sigma_ql':0.1})
p_bayes = {'length': 9,
                'noise': 0.01,
                'sigma': 0.1500005,
                'threshold': 1.0}
p_ql = {'alpha': 0.9, 
                'beta': 3.0, 
                'gamma': 0.2, 
                'sigma': 0.0900007}    

nb_trials = 39
nb_blocs = 10
cats = CATS(nb_trials)
models = dict({"fusion":FSelection(cats.states, cats.actions, p_fusion),
                    "qlearning":QLearning(cats.states, cats.actions, p_ql),
                    "bayesian":BayesianWorkingMemory(cats.states, cats.actions, p_bayes),
                    "selection":KSelection(cats.states, cats.actions, p_selection)})



for m in models.iterkeys():
    print m
    testModel(models[m])

# -----------------------------------
#order data
# -----------------------------------
data = dict()
data['pcr'] = dict()
for m in models.keys():
    data['pcr'][m] = extractStimulusPresentation(models[m].responses, models[m].state, models[m].action, models[m].responses)
# -----------------------------------
data['rt'] = dict()
for m in models.keys():    
    data['rt'][m] = dict()
    step, indice = getRepresentativeSteps(models[m].reaction, models[m].state, models[m].action, models[m].responses)
    data['rt'][m]['mean'], data['rt'][m]['sem'] = computeMeanRepresentativeSteps(step) 
    
# -----------------------------------


# -----------------------------------
# Plot
# -----------------------------------

fig = plt.figure()

dashes = {'fusion':'-','selection':'-*','bayesian':'-','qlearning':'-*'}

colors = ['blue','red','green']
line1 = tuple([plt.Line2D(range(2),range(2),linestyle=dashes[m],alpha=1.0,color='black') for m in ['bayesian', 'qlearning']])
plt.figlegend(line1,tuple(['BWM','Q-L']), loc = 'lower right', bbox_to_anchor = (0.79, 0.55))
line2 = tuple([plt.Line2D(range(2),range(2),linestyle=dashes[m],alpha=1.0,color='black') for m in ['fusion', 'selection']])
plt.figlegend(line2,tuple(['Entropy-based', 'VPI-based']), loc = 'lower right', bbox_to_anchor = (0.78, 0.20))

ax1 = fig.add_subplot(221)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()

for k in ['bayesian','qlearning']:
    m = data['pcr'][k]['mean']
    s = data['pcr'][k]['sem']
    for i in xrange(3):
        ax1.errorbar(range(1, len(m[i])+1), m[i], s[i], linestyle = dashes[k], color = colors[i], elinewidth = 2)
        ax1.plot(range(1, len(m[i])+1), m[i], linestyle = dashes[k], color = colors[i], linewidth = 2.5, label = 'Stim '+str(i+1))

ax1.set_ylabel("Probability correct responses")
ax1.set_xlabel("Trial")
fig.text(0.1, 0.92, "C.", fontsize = 22)
# legend(loc = 'lower right')
# xticks(range(2,11,2))
# xlabel("Trial")
ax1.set_xlim(0.8, 10.2)
ax1.set_ylim(-0.05, 1.05)
# yticks(np.arange(0, 1.2, 0.2))
# ylabel('Probability Correct Responses')
# title('A')

ax2 = fig.add_subplot(222)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()

for k in ['bayesian','qlearning']:
    m = data['rt'][k]['mean']
    s = data['rt'][k]['sem']
    ax2.errorbar(range(1, len(m)+1), m, s, color = 'black', marker = 'o', linewidth = 2.5, elinewidth = 2, linestyle = dashes[k])
    ax2.set_xlabel("Representative step")
    ax2.set_ylabel("Cycle")

ax2.set_ylim(0,10)

fig.text(0.5, 0.92, "D.", fontsize = 22)

ax1 = fig.add_subplot(223)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()

for k in ['fusion', 'selection']:
    m = data['pcr'][k]['mean']
    s = data['pcr'][k]['sem']
    for i in xrange(3):
        ax1.errorbar(range(1, len(m[i])+1), m[i], s[i], linestyle = dashes[k], color = colors[i], elinewidth = 2)
        ax1.plot(range(1, len(m[i])+1), m[i], linestyle = dashes[k], color = colors[i], linewidth = 2.5, label = 'Stim '+str(i+1))

ax1.set_ylabel("Probability correct responses")
ax1.set_xlabel("Trial")
ax1.set_xlim(0.8, 10.2)
ax1.set_ylim(-0.05, 1.05)
# legend(loc = 'lower right')
# xticks(range(2,11,2))
# xlabel("Trial")
# xlim(0.8, 10.2)
# ylim(-0.05, 1.05)
# yticks(np.arange(0, 1.2, 0.2))
# ylabel('Probability Correct Responses')
# title('A')

ax2 = fig.add_subplot(224)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()

for k in ['fusion', 'selection']:
    m = data['rt'][k]['mean']
    s = data['rt'][k]['sem']
    ax2.errorbar(range(1, len(m)+1), m, s, color = 'black', marker = 'o', linewidth = 2.5, elinewidth = 2, linestyle = dashes[k])
    ax2.set_xlabel("Representative step")
    ax2.set_ylabel("Cycle")
ax2.set_ylim(0,10)


subplots_adjust(hspace = 0.2)
fig.savefig(os.path.expanduser("~/Dropbox/ED3C/Journee_doctorant/poster/pics/beh_models.eps"), bbox_inches='tight')
os.system("evince "+os.path.expanduser("~/Dropbox/ED3C/Journee_doctorant/poster/pics/beh_models.eps"))

