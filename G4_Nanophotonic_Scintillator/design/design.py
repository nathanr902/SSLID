import time
from matplotlib import pyplot as plt
from hybrid_scintillator import *
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

pairs = 5
mu_Ti = 1.107e2
mu_Cl = 5.725e1
mu_C  = 2.373
mu_O  = 5.952
TiFractions = 0.30
OFractions = 0.4066
ClFractions = 0.0719
CFractions = 0.1290
density_sl = 1.3 + TiFractions/0.4 #g/cm^3
mu_gamma_sl = density_sl * 1e-4 * (mu_Ti * TiFractions + mu_O * OFractions + mu_Cl * ClFractions + mu_C * CFractions)
mu_gamma_scint = 1e-4 * 1.02 * (2.562e-1)
mu_electron_sl = 10.
mu_electron_scint = 8.
mu_sl_l = 0.4383617656171
mu_l_scint = 0.
C_sl = 1.
C_scint = 1.

μ = get_μ(mu_gamma_scint, mu_electron_scint, mu_l_scint, mu_gamma_sl, mu_electron_sl, mu_sl_l)
rates = ConversionRates(C_scint, C_sl, 1.)

scint_init_t = 1.
sl_init_t = 0.05

initial_scintillator_thicknesses = scint_init_t * torch.ones(pairs)
initial_sl_thicknesses = sl_init_t * torch.ones(pairs)
initial_thicknessses = torch.cat((initial_scintillator_thicknesses , initial_sl_thicknesses))
initial_thicknessses_rand = torch.cat((scint_init_t + 0.01 * torch.rand(pairs), sl_init_t + 0.01 * torch.rand(pairs)))

theory_N = lambda thicknesses: total_n_l(thicknesses[:pairs], thicknesses[pairs:], μ, rates, pairs)


class Model(nn.Module):

    def __init__(self, x0):
        super(Model, self).__init__()
        self.thicknesses = nn.Parameter(x0.clone())

    def forward(self):
        return theory_N(self.thicknesses)

hybrid_scintillator_model = Model(initial_thicknessses_rand)

optimizer = optim.Adam(hybrid_scintillator_model.parameters(), lr=0.2)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

ref = theory_N(initial_thicknessses).detach().numpy()

plot_optimization_process = True
if plot_optimization_process:
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(initial_sl_thicknesses, 'r', label='stopping layer thicknesses')
    ax.set_xlabel('SL thickness [um]')
    ax.set_ylabel('stopping layer', color='r')

    ax_twin = ax.twinx()
    ax_twin.set_ylabel('scintillator', color='b')
    line_twin, = ax_twin.plot(initial_scintillator_thicknesses, 'b', label='scintillator thicknesses')
    ax.set_ylim((0, 2))
    ax_twin.set_ylim((0, 2))

def barrier(x):
    if x.min() <= 0:
        return torch.inf
    else:
        return 1 / x.min()

losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    output = hybrid_scintillator_model()
    loss = -output# + barrier(hybrid_scintillator_model.thicknesses)
    losses.append(loss.detach().numpy())
    loss.backward()
    optimizer.step()
    scheduler.step()

    if plot_optimization_process:
        line.set_ydata(hybrid_scintillator_model.thicknesses[pairs:].detach().numpy())
        line_twin.set_ydata(hybrid_scintillator_model.thicknesses[:pairs].detach().numpy())
        enhancement = output.detach().numpy() / ref
        ax.set_title('iter ' + str(epoch) + ' enhancement = ' + str(enhancement))
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.01)

# bulk_light_yield = total_n_l(torch.Tensor([sum(initial_thicknessses[:pairs])]), torch.Tensor([0.001]), μ, rates, 1).detach().numpy()
bulk_light_yield = theory_N(initial_thicknessses).detach().numpy() / 2.85
print(torch.Tensor([sum(initial_thicknessses[:pairs])]), torch.Tensor([sum(initial_thicknessses[pairs:])]), bulk_light_yield)
enhancement_hetero = theory_N(initial_thicknessses).detach().numpy() / bulk_light_yield
enhancement_inverse = theory_N(hybrid_scintillator_model.thicknesses).detach().numpy() / bulk_light_yield
print('enhancements:', enhancement_hetero, enhancement_inverse)

labels = ['Uniform scintillator', 'Multilayer', 'Inverse design']
values = [1., enhancement_hetero, enhancement_inverse]

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)


exp_color = (137/255, 165/255, 84/255)
theo_color = (177/255, 93/255, 91/255)
sim_color = (56/255, 104/255, 108/255)
line_width = 3
ax.bar([1,2,3], values, color=theo_color)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(line_width)
ax.tick_params(width=line_width) 
ax.set_xlabel('SL thickness [um]')
ax.set_ylabel('relative photons count')
ax.set_xticks([1,2,3])
ax.set_ylim(0,6)
# Add title and labels for x and y axis
plt.title('Bar Plot Example')
plt.xlabel('Labels')
plt.ylabel('Values')
plt.savefig('/home/vboxuser/Projects/G4NS/G4_Nanophotonic_Scintillator/analysis/inverse_desgin.png', dpi=330)
plt.show()