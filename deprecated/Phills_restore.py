import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PINNs import RSTnet
from Utilities import plot_contour, plot_streamlines, plot_lines, relative_l2

tf.compat.v1.enable_eager_execution()


invRe = 1 / 5600
#############################################################
########################     Data Loading    ################
#############################################################
domain_data = np.load("Data/domain_data.npy")
domain_in = np.load("Data/domain_in.npy")
bc_data = np.load("Data/bc_data.npy")
bc_in = np.load("Data/bc_in.npy")


#############################################################
########################      Data Preparation        #######
#############################################################
N = domain_data.shape[0]
x = domain_in[:, 0:1]
y = domain_in[:, 1:2]
u = domain_data[:, 0:1]
v = domain_data[:, 1:2]
p = domain_data[:, 2:3]
uu = domain_data[:, 3:4]
uv = domain_data[:, 4:5]
vv = domain_data[:, 5:6]
train_pi = np.concatenate([x, y], axis=1)
print("domain data dim:", train_pi.shape[0])

Nbc = bc_data.shape[0]
xbc = bc_in[:, 0:1]
ybc = bc_in[:, 1:2]
ubc = bc_data[:, 0:1]
vbc = bc_data[:, 1:2]
pbc = bc_data[:, 2:3]
uubc = bc_data[:, 3:4]
uvbc = bc_data[:, 4:5]
vvbc = bc_data[:, 5:6]
train_dd = np.concatenate([xbc, ybc, ubc, vbc, pbc, uubc, uvbc, vvbc], axis=1)
print("bc data dim:", train_dd.shape[0])

np.random.shuffle(train_pi)
np.random.shuffle(train_dd)

############################################################
########################   Model Constrcution        #######
############################################################
layers_u = [2, 64, 64, 64, 64, 64, 64, 64, 64, 9]
lr = 0.001
if_positivity = [True, "softplus"]  # if True, choose between softplus and relu
if_realisability = False
model = RSTnet(
    invRe,
    train_dd,
    train_pi,
    layers_u,
    lr,
    if_positivity=if_positivity,
    if_realisability=if_realisability,
)

#################################################################
########################   Model Restore         ################
#################################################################

model.restore(path="Model-lbfgs/")

###########################################################
########################   Plot        ####################
###########################################################
u_star, v_star, p_star, uu_star, uv_star, vv_star = (
    domain_data[:, 0],
    domain_data[:, 1],
    domain_data[:, 2],
    domain_data[:, 3],
    domain_data[:, 4],
    domain_data[:, 5],
)
x_star, y_star = domain_in[:, 0:1], domain_in[:, 1:2]
u_pred, v_pred, p_pred, uu_pred, uv_pred, vv_pred = model.predict(x_star, y_star)


fig, axs = plt.subplots(2, 3, figsize=(10, 4.5), dpi=300)

plot_contour(
    axs[0, 0],
    x_star.flatten(),
    y_star.flatten(),
    u_pred.flatten(),
    levels=np.linspace(-0.4, 1.2, 81),
)
axs[0, 0].set_aspect(1)
axs[0, 0].set_title("RSTnet:U")
plot_contour(
    axs[0, 1],
    x_star.flatten(),
    y_star.flatten(),
    v_pred.flatten(),
    levels=np.linspace(-0.15, 0.15, 61),
)
axs[0, 1].set_aspect(1)
axs[0, 1].set_title("RSTnet:V")
plot_streamlines(
    axs[0, 2],
    x_star.flatten(),
    y_star.flatten(),
    u_pred.flatten(),
    v_pred.flatten(),
    levels=np.linspace(-0.15, 0.15, 61),
)
axs[0, 2].set_aspect(1)
axs[0, 2].set_title("RSTnet:Streamlines")

plot_contour(
    axs[1, 0],
    x_star.flatten(),
    y_star.flatten(),
    u_star.flatten(),
    levels=np.linspace(-0.4, 1.2, 81),
)
axs[1, 0].set_aspect(1)
axs[1, 0].set_title("DNS:U")
plot_contour(
    axs[1, 1],
    x_star.flatten(),
    y_star.flatten(),
    v_star.flatten(),
    levels=np.linspace(-0.15, 0.15, 61),
)
axs[1, 1].set_aspect(1)
axs[1, 1].set_title("DNS:V")
plot_streamlines(
    axs[1, 2],
    x_star.flatten(),
    y_star.flatten(),
    u_star.flatten(),
    v_star.flatten(),
    levels=np.linspace(-0.15, 0.15, 61),
)
axs[1, 2].set_aspect(1)
axs[1, 2].set_title("DNS:Streamlines")
plt.savefig("Results/uvstream_hills.png")


fig, axs = plt.subplots(2, 3, figsize=(10, 4.5), dpi=300)

plot_contour(
    axs[0, 0],
    x_star.flatten(),
    y_star.flatten(),
    uu_pred.flatten(),
    levels=np.linspace(0, 0.1, 51),
)
axs[0, 0].set_aspect(1)
axs[0, 0].set_title(r"RSTnet:$\overline{uu}$")
plot_contour(
    axs[0, 1],
    x_star.flatten(),
    y_star.flatten(),
    uv_pred.flatten(),
    levels=np.linspace(-0.05, 0.01, 61),
)
axs[0, 1].set_aspect(1)
axs[0, 1].set_title(r"RSTnet:$\overline{uv}$")
plot_contour(
    axs[0, 2],
    x_star.flatten(),
    y_star.flatten(),
    vv_pred.flatten(),
    levels=np.linspace(0.0, 0.06, 61),
)
axs[0, 2].set_aspect(1)
axs[0, 2].set_title(r"RSTnet:$\overline{vv}$")

plot_contour(
    axs[1, 0],
    x_star.flatten(),
    y_star.flatten(),
    uu_star.flatten(),
    levels=np.linspace(0, 0.1, 51),
)
axs[1, 0].set_aspect(1)
axs[1, 0].set_title(r"DNS:$\overline{uu}$")
plot_contour(
    axs[1, 1],
    x_star.flatten(),
    y_star.flatten(),
    uv_star.flatten(),
    levels=np.linspace(-0.05, 0.01, 61),
)
axs[1, 1].set_aspect(1)
axs[1, 1].set_title(r"DNS:$\overline{uv}$")
plot_contour(
    axs[1, 2],
    x_star.flatten(),
    y_star.flatten(),
    vv_star.flatten(),
    levels=np.linspace(0.0, 0.06, 61),
)
axs[1, 2].set_aspect(1)
axs[1, 2].set_title(r"DNS:$\overline{vv}$")
plt.savefig("Results/uuuvvv_hills.png")

###########################################################
########################   Line Plot        ###############
###########################################################
alpha = 1.0

fig, axs = plt.subplots(1, 2, figsize=(8, 2.5), dpi=300)
plot_lines(
    axs[0],
    x_star.flatten(),
    y_star.flatten(),
    2 * u_pred.flatten(),
    2 * u_star.flatten(),
    alpha=alpha,
    legend_loc="upper left",
    bbox_to_anchor=(0.02, 0.9),
)
axs[0].set_xlabel(r"x/H; 2U/$U_b$+x/H")
plot_lines(
    axs[1],
    x_star.flatten(),
    y_star.flatten(),
    25 * uv_pred.flatten(),
    25 * uv_star.flatten(),
    alpha=alpha,
    legend_loc="upper right",
    bbox_to_anchor=(0.9, 0.9),
)
axs[1].set_xlabel(r"x/H; 25$\overline{uv}$/$U_b^2$+x/H")
plt.savefig("Results/Uuvlines_hills.png")


###########################################################
########################   Error        ###################
###########################################################
print("#########################################")
print("U l2:", relative_l2(u_star.flatten(), u_pred.flatten()))
print("V l2:", relative_l2(v_star.flatten(), v_pred.flatten()))
print("uu l2:", relative_l2(uu_star.flatten(), uu_pred.flatten()))
print("uv l2:", relative_l2(uv_star.flatten(), uv_pred.flatten()))
print("vv l2:", relative_l2(vv_star.flatten(), vv_pred.flatten()))
print("#########################################")
