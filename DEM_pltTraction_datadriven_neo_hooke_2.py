import matplotlib
import matplotlib.pyplot as plt
# Look at compatibility

import torch
import torch.nn as nn
import numpy as np

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8) # Use _maximally_ 8 CPU cores
plt.rcParams.update({'font.size': 16})

device = torch.device("cpu")
#device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
device = torch.device(device)


Lx = 2
Ly = 1
samples_x = 20 * Lx + 1
samples_y = 20 * Ly + 1
delta_x = Lx / (samples_x - 1)
delta_y = Ly / (samples_y - 1)

disp_left = 0
force_right = 0.2
force_upper = 0
force_lower = 0

increments = 1

"""
/|-----------------------| ->
/|                       | ->
/|                       | ->
/|-----------------------| ->
"""

sample_points = torch.meshgrid(torch.linspace(0, Lx, samples_x), torch.linspace(0, Ly, samples_y), indexing='ij')
sample_points = torch.cat((sample_points[0].reshape(-1, 1), sample_points[1].reshape(-1, 1)), dim=1)
sample_points = sample_points.to(device)
sample_points.requires_grad_(True)


hidden_dim = 16
input_dim = 2
output_dim = 2


def plot_undef():
    plt.subplot(1,1,1)
    plt.scatter(sample_points[:, 0].detach().numpy(), sample_points[:, 1].detach().numpy(), c='black', s=5)

E = 1
nu = 0.3
mu = E/(2*(1+nu))
lam = E*nu / ((1+nu)*(1-2*nu))

def get_F(X, U):
    duxdxy = torch.autograd.grad(U[:, 0].unsqueeze(1), X, torch.ones(X.size()[0], 1, device=device),
                                 create_graph=True, retain_graph=True)[0]
    duydxy = torch.autograd.grad(U[:, 1].unsqueeze(1), X, torch.ones(X.size()[0], 1, device=device),
                                 create_graph=True, retain_graph=True)[0]
    H = torch.zeros(X.size()[0], X.size()[1], X.size()[1], device=device)
    H[:, 0, :] = duxdxy
    H[:, 1, :] = duydxy
    H = H.reshape(samples_x, samples_y, 2, 2)

    F = H + torch.eye(2)
    return F

def simpson_integration(f, dx, dim):
    n = f.size()[dim]
    c = torch.zeros(f.size(dim))
    c[0] = 1
    c[-1] = 1
    c[1:-1] = 2
    c[1:-1:2] += 2
    S = []
    h = dx/3
    S = torch.tensordot(f,c, dims = ([dim], [0]))
    return h*S
    
def material_model(F):
    right_CG = torch.einsum('bijk,bikj->bijk', F, F)
    J = torch.det(F)
    tr_C = right_CG.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
    F_T_inv = torch.inverse(F.transpose(-2, -1))
    term1 = mu * (F - F_T_inv)
    term2 = lam* torch.mul((J[:,:,None, None]-1)*J[:,:,None,None], F_T_inv)
    term12 = (1/J[:,:,None, None]) *(term1 + term2)
    sig = torch.mul(term12, torch.transpose(F,-1,-2))
    psi = (mu / 2) * (tr_C - 2 - 2 * torch.log(J)) + (lam / 2) * (torch.log(J) ** 2)

    #P = torch.autograd.grad(psi, F, torch.ones(F.size()[0], F.size()[1], device=device),create_graph=True, retain_graph=True)[0]
    
    return psi, sig


def integratePsi(psi):
    # PyTorch has a built-in function for trapezoidal integration.
    # Hence, we can use it (nested for the 2 dimensions)
    # The comparison with midpoint rule showed no
    # difference to the midpoint rule Pnificant for the basic stability of this DEM code
    PSI = torch.trapezoid(torch.trapezoid(psi, dx = delta_y, dim=1), dx = delta_x, dim=0)
    return PSI


def calculateDivergenceLoss(sig):
    criterion = nn.MSELoss(reduction="sum")
    div_sig = (1 / (2 * delta_x) *
               (sig[2:, 1:-1, :, 0] - sig[:-2, 1:-1, :, 0]))
    div_sig += (1 / (2 * delta_y) *
                (sig[1:-1, 2:, :, 1] - sig[1:-1, :-2, :, 1]))
    div_loss = criterion(div_sig, torch.zeros(
        samples_x - 2, samples_y - 2, 2, device=device))
    return div_loss


def integrateTractionEnergy(U, sig):
    # referred to the tractions, not forces
    # Again, use built-in integration
    # Before, there was a bug: Only the farmost right points have to be integrated
    # Not U[:, :, 0], but U[-1, :, 0]
    # This bug is fixed here:
    Tx = force_right * simpson_integration(U[-1,:,0], dx=delta_y, dim=0)
    #Tx = force_right * torch.trapezoid(U[-1,:,0], dx=delta_y)
    #Tx = torch.trapezoid(P[-1, :, 0, 0] * U[-1, :, 0], dx=delta_y)
    #Ty = torch.trapezoid(sig[-1, :, 0, 1] * U[-1, :, 1], dx=delta_y)
    Ty = simpson_integration(sig[-1, :, 0, 1] * U[-1, :, 1], dx=delta_y, dim=0)
    # Theoretically, the traction is calculated from the stress, not the force boundary condition
    # So it's up to comparison what performs better.
    # Ty = torch.trapezoid(P[-1, :, 0, 0] * U[-1, :, 0], dx=delta_y)

    # The x-component was not necessary, so far
    # Careful: the bug was not fixed here!
    # Tx = (P[1:-1, :, 0, 0] * U[1:-1, :, 0]).sum()
    # Tx += (P[:, 1:-1, 1, 1] * U[:, 1:-1, 1]).sum()
    # Tx += 0.5 * (P[[0, -1], [0, -1], 1, 1] * U[[0, -1], [0, -1], 1]).sum()
    # Tx *= delta_x

    return (Ty, Tx)


def integrateTractionOther(U, sig):
    # not multiplicate for -, because we use absolute reference system
    #Tx_dwn = torch.trapezoid(sig[:, 0, 1, 0] * U[:, 0, 0], dx=delta_x)
    #Ty_dwn = torch.trapezoid(sig[:, 0, 1, 1] * U[:, 0, 1], dx=delta_x)

    #Tx_up = torch.trapezoid(sig[:, -1, 1, 0] * U[:, -1, 0], dx=delta_x)
    #Ty_up = torch.trapezoid(sig[:, -1, 1, 1] * U[:, -1, 1], dx=delta_x)
    
    Tx_dwn = simpson_integration(sig[:, 0, 1, 0] * U[:, 0, 0], dx=delta_x, dim=0)
    Ty_dwn = simpson_integration(sig[:, 0, 1, 1] * U[:, 0, 1], dx=delta_x, dim=0)

    Tx_up = simpson_integration(sig[:, -1, 1, 0] * U[:, -1, 0], dx=delta_x, dim=0)
    Ty_up = simpson_integration(sig[:, -1, 1, 1] * U[:, -1, 1], dx=delta_x, dim=0)

    return (Tx_dwn, Ty_dwn, Tx_up, Ty_up)


def boundaryLosses(U, sig):
    criterion = nn.MSELoss(reduction="sum")
    # New factor to multiply the losses for fixed / Dirichlet B.C.
    # The factor 100*E was too large, i.e. the other conditions were not respected in comparison.
    # The factor 1*E was too small, i.e. the B.C. was not fulfilled well
    loss_left = criterion(
        delta_y * U[0, :, :] * E*10, torch.zeros(samples_y, 2, device=device))

    # no fundamental changes
    forces_right = torch.ones(samples_y, device=device)
    forces_right[[0, -1]] = 0.5 * torch.ones(2, device=device)
    forces_right *= force_right * delta_y/Ly

    # Factor of 10 introduced to improve the compliance with this B.C.
    tractions_right = sig[-1, :, 0, 0] * delta_y
    loss_right = 2*criterion(tractions_right, forces_right)

    # no changes
    sheartractions_right = sig[-1, :, 0, 1] * delta_y
    loss_right += criterion(sheartractions_right,
                            torch.zeros(samples_y, device=device))
    tractions_lower = sig[1:-1, 0, 1, 1] * delta_x
    loss_lower = criterion(tractions_lower, torch.zeros(
        samples_x - 2, device=device))
    sheartractions_lower = sig[1:-1, 0, 1, 0] * delta_x
    loss_lower += criterion(sheartractions_lower,
                            torch.zeros(samples_x - 2, device=device))
    tractions_upper = sig[1:-1, -1, 1, 1] * delta_x
    loss_upper = criterion(tractions_upper, torch.zeros(
        samples_x - 2, device=device))
    sheartractions_upper = sig[1:-1, -1, 1, 0] * delta_x
    loss_upper += criterion(sheartractions_upper,
                            torch.zeros(samples_x - 2, device=device))
    loss_upper_lower = criterion(tractions_upper,-tractions_lower)
    bc_losses = loss_right  + loss_left + loss_upper_lower #+ loss_upper + loss_lower #loss_right + 
    return bc_losses


def calc_losses(U, plot, data_driven):
    F = get_F(sample_points, U)
    psi, sig = material_model(F)

    # Cool plots to better understand what's going on
    if plot:
        plt.figure(figsize=(25, 25))
        plt.subplot(7, 3, 1)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U[:, 1]).detach().cpu().numpy(),
                    c=(F[:, :, 0, 0].detach().cpu().numpy()))
        plt.colorbar()
        plt.title("eps_x")
        plt.subplot(7, 3, 2)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U[:, 1]).detach().cpu().numpy(),
                    c=(F[:, :, 1, 1].detach().cpu().numpy()))
        plt.colorbar()
        plt.title("eps_y")
        plt.subplot(7, 3, 3)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U[:, 1]).detach().cpu().numpy(),
                    c=(F[:, :, 0, 1].detach().cpu().numpy()))
        plt.title("eps_xy")
        plt.colorbar()
        plt.subplot(7, 3, 4)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U[:, 1]).detach().cpu().numpy(),
                    c=(psi[:, :].detach().cpu().numpy()))
        plt.colorbar()
        plt.title("psi")

        plt.subplot(7, 3, 7)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U[:, 1]).detach().cpu().numpy(),
                    c=(U[:, 0].detach().cpu().numpy()))
        plt.title("U_x")
        plt.colorbar()
        plt.subplot(7, 3, 8)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U[:, 1]).detach().cpu().numpy(),
                    c=(U[:, 1].detach().cpu().numpy()))
        plt.title("U_y")
        plt.colorbar()

        plt.tight_layout()

    U = U.reshape(samples_x, samples_y, 2)
    #PSI = integratePsi(psi)
    PSI = simpson_integration(simpson_integration(psi, delta_x, 1), delta_x, 0)
    #print(PSI, simpson_PSI)
    Ty, Tx = integrateTractionEnergy(U, sig)
    bc_losses = boundaryLosses(U, sig)
    div_losses = calculateDivergenceLoss(sig)
    T_others = integrateTractionOther(U, sig)

    # New: calculate an _approximated_ analytical solution
    # Lz = 1
    # (Actually, the lateral shrinkage is not linear!)
    # It can be used for a data-driven training of the NN --> see, if the NN architecture
    # is qualitatively capable of approximating the solution.
    # It also helped to find bugs if the NN was trained data-driven first and then physics-informed, afterwards.
    eps_an = force_right/E*Ly*1
    delta_l = eps_an*Lx
    U_target = torch.zeros_like(sample_points)
    U_target[:, 0] = sample_points[:, 0]/Lx * delta_l
    U_target[:, 1] = -sample_points[:, 0]/Lx * nu * \
        (sample_points[:, 1] / Ly - 0.5) * delta_l
    U_grid = U_target.reshape(samples_x, samples_y, 2)

    T = Tx
    forces_right = torch.ones(samples_y, device=device)
    forces_right[[0, -1]] = 0.5 * torch.ones(2, device=device)
    forces_right *= force_right * delta_y

    T_an = np.array(
        [torch.sum(forces_right * U_grid[-1, :, 0], dim=0).item(), 0])

    T_nn = np.array([Tx.item(), Ty.item()])

    diff_T = T_an - T_nn

    F = get_F(sample_points, U_target)
    psi, _ = material_model(F)
    PSI_an = integratePsi(psi)

    # Plot the _approximated solution_
    if plot:
        plt.subplot(7, 3, 10)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U_target[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U_target[:, 1]
                     ).detach().cpu().numpy(),
                    c='black')
        plt.title("deformed")

        plt.subplot(7, 3, 13)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U_target[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U_target[:, 1]
                     ).detach().cpu().numpy(),
                    c=(F[:, :, 0, 0].detach().cpu().numpy()))
        plt.colorbar()
        plt.title("eps_x")

        plt.subplot(7, 3, 14)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U_target[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U_target[:, 1]
                     ).detach().cpu().numpy(),
                    c=(F[:, :, 1, 1].detach().cpu().numpy()))
        plt.colorbar()
        plt.title("eps_y")

        plt.subplot(7, 3, 15)
        plt.gca().set_aspect('equal')
        plt.scatter((sample_points[:, 0] + U_target[:, 0]).detach().cpu().numpy(),
                    (sample_points[:, 1] + U_target[:, 1]
                     ).detach().cpu().numpy(),
                    c=(F[:, :, 0, 1].detach().cpu().numpy()))
        plt.colorbar()
        plt.title("eps_xy")
        plt.tight_layout()

        Ux_mean_an = torch.mean(U_grid[:,:,0], dim=1)
        Ux_mean_NN = torch.mean(U[:,:,0], dim=1)

        plt.figure()
        plt.plot((torch.unique(sample_points[:, 0])).detach().cpu().numpy(),
                    Ux_mean_an.detach().cpu().numpy(), label='analytic')
        plt.plot((torch.unique(sample_points[:, 0])).detach().cpu().numpy(),
                    Ux_mean_NN.detach().cpu().numpy(), label='DEM')
        plt.title("$U_x$")
        plt.legend()
        plt.show()


        plt.show()

    # For data-driven training
    U_target = U_target.reshape(samples_x, samples_y, 2)
    criterion = nn.MSELoss(reduction="sum")

    if data_driven:
        print("data_driven")
        return (criterion(U, U_target),
                (T - PSI).pow(2), bc_losses, div_losses, T_others, diff_T, PSI, PSI_an)
    else:
        return (0.95*(T-PSI).pow(2) + 1.5* bc_losses + 0.1 * div_losses,
                (T-PSI).pow(2), bc_losses, div_losses, T_others, diff_T, PSI, PSI_an)


# This allows to choose another (random) initialization for the NN parameters
def weights_init(m):
    if isinstance(m, nn.Linear):
        #torch.nn.init.uniform_(m.weight, -1, 1)
        torch.nn.init.normal_(m.weight, mean=1.0/hidden_dim, std=10*1.0/hidden_dim)
        #torch.nn.init.zeros_(m.weight)
        torch.nn.init.zeros_(m.bias)

# This is a custom activation function, that even has an own NN parameter (self.slope)
class myPow(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.pow(x, 2) + self.slope * x

class MLNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLNet, self).__init__()
        self.hidden_dim = hidden_dim

        self.fcnn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            #nn.Tanh(),
            myPow(), #--> so easy is it to use a custom activation function
            # Which one is better? Try it out!
            nn.Linear(hidden_dim, output_dim, bias=True),
        )
        #self.fcnn1.apply(weights_init) --> comment in to apply the custom (random) intialization function

    def forward(self, x):
        #x_in = torch.cat((x[:, 0].unsqueeze(1)/Lx, x[:, 1].unsqueeze(1)/Ly - 0.5), dim=1) -> Normalize inputs, try it out!
        x_in = x
        out = force_right/(E*1/Lx) * self.fcnn1(x_in)
        #out = self.fcnn1(x)
        out[:, 0] *= x[:, 0]
        out[:, 1] *= x[:, 0] #--> In the literature a common output transformation, not proven to be necessary, here
        # In fact, that includes problem specific information and obstructs generalizability of the method
        # to a certain extent
        return out

# Should the training start data-driven with the _approximated_ solution as target values?
data_driven = True

def train():
    model = MLNet(input_dim, hidden_dim, output_dim)
    model.to(device)
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=20)#, line_search_fn='strong_wolfe')
    # Adam has given much better results than LBFGS during the test runs for this sourcecode version.
    # lr=0.01 is not optimized, but this order of magnitude performed comparatively well.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)# 0.001 
    

    losses = []
    T_psis = []
    bound_losses = []
    div_losses = []
    T_others = []
    diff_Ts = []
    PSIs = []
    PSIs_an = []
    force_right_total = 0.2
    for increment in range(increments):
        force_right = (increment+1)/increments*force_right_total
        epochs = 1000
        for epoch in range(epochs):
            model.train()
        #if (epoch + 1) % 3 == 0:
        #    global force_right
        #    if force_right < 0.05:
        #        force_right += 0.03
            if increment > 0 or epoch > 500:
                global data_driven
                data_driven = False # --> the following epochs will be a kind of transfer learning
            if (epoch) % 1000 == 0:
                U = model(sample_points)
                print("force_right: ", force_right, "increment:", increment)
                calc_losses(U, True, True)
                plt.show()

            def closure():
                U = model(sample_points)
                global data_driven
                loss, T_psi, bound_loss, div_loss, T_others, diff_T, PSI, PSI_an = calc_losses(U, False, data_driven)
                print('Epoch %i/%i, Total Loss: %.64e' % (epoch+1, epochs, loss.item()))
                optimizer.zero_grad()
                loss.backward()
                return (loss, T_psi, bound_loss, div_loss, T_others, diff_T, PSI, PSI_an)

            loss, T_psi, bound_loss, div_loss, T_other, diff_T, PSI, PSI_an = closure()

            losses.append(loss.detach().numpy())
            T_psis.append(T_psi.detach().numpy())
            bound_losses.append(bound_loss.detach().numpy())
            div_losses.append(div_loss.detach().numpy())
            T_others.append(tuple(element.detach().item() for element in T_other))
            diff_Ts.append(diff_T)
            PSIs.append(PSI.detach().numpy())
            PSIs_an.append(PSI_an.detach().numpy())

            optimizer.step()
        #optimizer.step(closure)

    U = model(sample_points)
    calc_losses(U, True, data_driven)
    losses = np.array(losses)
    T_psis = np.array(T_psis)
    bound_losses = np.array(bound_losses)
    div_losses = np.array(div_losses)
    PSIs = np.array(PSIs)
    PSIs_an = np.array(PSIs_an)
    PSIs = np.stack([PSIs_an, PSIs], axis=-1)

    return (model, losses, T_psis, bound_losses, div_losses, T_others, diff_Ts, PSIs)

def plot_losses(losses: tuple):
    plt.figure()
    epochs = np.arange(0, len(losses[0]))

    plt.plot(epochs, losses[0], label='Residual loss')
    plt.plot(epochs, losses[1], label='T-PSI')
    plt.plot(epochs, losses[2], label='boundary loss')
    plt.plot(epochs, losses[3], label='divergence loss')

    plt.xlabel('epoch')
    plt.yscale('log')
    plt.legend()
    plt.show()

def tractions_to_arr(T_others: list, diff_Ts: list):
    Tx_down = []
    Ty_down = []
    Tx_up = []
    Ty_up = []
    Tx_right = []
    Ty_right = []

    for tractionOther in T_others:
        Tx_down.append(tractionOther[0])
        Ty_down.append(tractionOther[1])
        Tx_up.append(tractionOther[2])
        Ty_up.append(tractionOther[3])

    for traction in diff_Ts:
        Tx_right.append(traction[0])
        Ty_right.append(traction[1])

    Tx_down = np.array(Tx_down)
    Ty_down = np.array(Ty_down)
    Tx_up = np.array(Tx_up)
    Ty_up = np.array(Ty_up)
    Tx_right = np.array(Tx_right)
    Ty_right = np.array(Ty_right)

    return (Tx_down, Ty_down, Tx_up, Ty_up, Tx_right, Ty_right)

def plotTractions(Ts: tuple):
    epochs = np.arange(0, len(Ts[0]))

    plt.figure()
    plt.plot(epochs, Ts[0], label = 'transversal component')
    plt.plot(epochs, Ts[1], label = 'normal component')
    plt.title("$T_{an} - T_{NN}$ - lower face")
    plt.xlabel('epochs')
    plt.legend()

    plt.figure()
    plt.plot(epochs, Ts[2], label = 'transversal component')
    plt.plot(epochs, Ts[3], label = 'normal component')
    plt.title("$T_{an} - T_{NN}$ - upper face")
    plt.xlabel('epochs')
    plt.legend()

    plt.figure()
    plt.plot(epochs, Ts[4], label = 'normal component')
    plt.plot(epochs, Ts[5], label = 'transversal component')
    plt.title("$T_{an} - T_{NN}$ - right face")
    plt.xlabel('epochs')
    plt.legend()

    plt.show()

def plotPSIs(PSIs: np.ndarray):
    epochs = np.arange(0, PSIs.shape[0])
    print(PSIs[0,0])

    plt.figure()
    plt.plot(epochs, PSIs[:,0], label='$\\Psi_{{an}}$')
    plt.plot(epochs, PSIs[:,1], label='$\\Psi_{{nn}}$')
    plt.plot(epochs, 0.02*np.ones((len(epochs),)), label='$\\Psi_{{FEM}}$')
    plt.title("$\\Psi$")
    plt.xlabel('epochs')
    plt.legend()

    plt.show()


model_trained, losses, res_losses, boundary_losses, div_losses, T_others, diff_Ts, PSIs = train()
losses = (losses, res_losses, boundary_losses, div_losses)
plot_losses(losses)
Ts = tractions_to_arr(T_others, diff_Ts)
plotTractions(Ts)
plotPSIs(PSIs)

