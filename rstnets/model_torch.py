import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import CaseConfig
from .networks_torch import FCNN


def _grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        retain_graph=True, create_graph=True,
    )[0]


class RSTnet:
    """Torch counterpart to `rstnets.model.RSTnet` with the same public method surface."""

    def __init__(self, case_config: CaseConfig, train_dd, train_pi, Lxscale=None, Lyscale=None,
                 precision=None, device=None):
        self.invRe = case_config.invRe
        self.layers_u = case_config.network.layers
        self.lr = case_config.network.learning_rate
        self.if_positivity = case_config.if_positivity
        self.if_realisability = case_config.if_realisability
        self.use_coord_scaling = case_config.use_coord_scaling
        self.supervise_pressure = case_config.supervise_pressure

        precision = precision or case_config.precision
        self.dtype = torch.float64 if precision == "float64" else torch.float32
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        if self.use_coord_scaling and (Lxscale is None or Lyscale is None):
            raise ValueError(f"case {case_config.name!r} requires coordinate scaling but "
                              "Lxscale/Lyscale were not provided")
        self.Lxscale = Lxscale
        self.Lyscale = Lyscale

        def to_t(a):
            return torch.as_tensor(a, dtype=self.dtype, device=self.device)

        x, y = train_dd[:, 0:1], train_dd[:, 1:2]
        u, v, p, uu, uv, vv = (train_dd[:, i:i + 1] for i in range(2, 8))
        x_pi, y_pi = train_pi[:, 0:1], train_pi[:, 1:2]

        self.x, self.y = to_t(x), to_t(y)
        self.u, self.v, self.p = to_t(u), to_t(v), to_t(p)
        self.uu, self.uv, self.vv = to_t(uu), to_t(uv), to_t(vv)
        self.x_pi, self.y_pi = to_t(x_pi), to_t(y_pi)

    def net_RANS(self, x, y):
        x = x.clone().requires_grad_(True)
        y = y.clone().requires_grad_(True)
        if self.use_coord_scaling:
            x_s = x / self.Lxscale
            y_s = y / self.Lyscale
        else:
            x_s, y_s = x, y

        utauf = self.u_model(torch.cat([x_s, y_s], dim=-1))
        u = utauf[:, 0:1]
        v = utauf[:, 1:2]
        p = utauf[:, 2:3]
        if self.if_positivity[0] is False:
            tau1 = utauf[:, 3:4]
            tau3 = utauf[:, 5:6]
        else:
            if self.if_positivity[1] == 'softplus':
                tau1 = F.softplus(utauf[:, 3:4])
                tau3 = F.softplus(utauf[:, 5:6])
            elif self.if_positivity[1] == 'relu':
                tau1 = F.relu(utauf[:, 3:4])
                tau3 = F.relu(utauf[:, 5:6])
            else:
                raise ValueError(f"unsupported if_positivity mode: {self.if_positivity[1]!r}")

        if self.if_realisability is False:
            tau2 = utauf[:, 4:5]
        else:
            tau2 = torch.sqrt(tau1) * torch.sqrt(tau3) * torch.tanh(utauf[:, 4:5])

        ftau1 = utauf[:, 6:7]
        ftau2 = utauf[:, 7:8]
        ftau3 = utauf[:, 8:9]

        u_x = _grad(u, x)
        u_y = _grad(u, y)
        u_xx = _grad(u_x, x)
        u_yy = _grad(u_y, y)
        v_x = _grad(v, x)
        v_y = _grad(v, y)
        v_xx = _grad(v_x, x)
        v_yy = _grad(v_y, y)
        p_x = _grad(p, x)
        p_y = _grad(p, y)

        tau1_x = _grad(tau1, x)
        tau1_y = _grad(tau1, y)
        tau2_x = _grad(tau2, x)
        tau2_y = _grad(tau2, y)
        tau3_x = _grad(tau3, x)
        tau3_y = _grad(tau3, y)

        f_div = u_x + v_y
        f_u = (u * u_x + v * u_y) + p_x - self.invRe * (u_xx + u_yy) + tau1_x + tau2_y
        f_v = (u * v_x + v * v_y) + p_y - self.invRe * (v_xx + v_yy) + tau2_x + tau3_y
        f_tau1 = (u * tau1_x + v * tau1_y) + (2 * tau1 * u_x + 2 * tau2 * u_y) + ftau1
        f_tau2 = (u * tau2_x + v * tau2_y) + (tau1 * v_x + tau2 * v_y + tau2 * u_x + tau3 * u_y) + ftau2
        f_tau3 = (u * tau3_x + v * tau3_y) + (2 * tau2 * v_x + 2 * tau3 * v_y) + ftau3

        return u, v, p, tau1, tau2, tau3, f_div, f_u, f_v, f_tau1, f_tau2, f_tau3

    def _loss_groups(self, x, y, u, v, p, uu, uv, vv, x_pi, y_pi):
        u_pred, v_pred, p_pred, uu_pred, uv_pred, vv_pred, _, _, _, _, _, _ = self.net_RANS(x, y)
        _, _, _, _, _, _, f_div, f_u_pred, f_v_pred, f_tau1_pred, f_tau2_pred, f_tau3_pred = self.net_RANS(x_pi, y_pi)

        group_uvbc = torch.mean((u - u_pred) ** 2) + torch.mean((v - v_pred) ** 2)
        if self.supervise_pressure:
            group_uvbc = group_uvbc + torch.mean((p - p_pred) ** 2)
        group_stressbc = (torch.mean((uu - uu_pred) ** 2) +
                           torch.mean((uv - uv_pred) ** 2) +
                           torch.mean((vv - vv_pred) ** 2))
        group_momentum = (torch.mean(f_div ** 2) +
                           torch.mean(f_u_pred ** 2) +
                           torch.mean(f_v_pred ** 2))
        group_stresstransport = (torch.mean(f_tau1_pred ** 2) +
                                  torch.mean(f_tau2_pred ** 2) +
                                  torch.mean(f_tau3_pred ** 2))
        return group_uvbc, group_stressbc, group_momentum, group_stresstransport

    def grad(self, x, y, u, v, p, uu, uv, vv, x_pi, y_pi):
        group_uvbc, group_stressbc, group_momentum, group_stresstransport = self._loss_groups(
            x, y, u, v, p, uu, uv, vv, x_pi, y_pi)
        loss = group_uvbc + group_stressbc + group_momentum + group_stresstransport
        grads = torch.autograd.grad(loss, list(self.u_model.parameters()))
        return loss.detach(), grads

    def monitor(self, x, y, u, v, p, uu, uv, vv, x_pi, y_pi):
        with torch.enable_grad():
            group_uvbc, group_stressbc, group_momentum, group_stresstransport = self._loss_groups(
                x, y, u, v, p, uu, uv, vv, x_pi, y_pi)
        return group_uvbc.item(), group_stressbc.item(), group_momentum.item(), group_stresstransport.item()

    def train(self, n_iters, batch_data, batch_pi, history_path, print_interval=100):
        self.u_model = FCNN(self.layers_u).to(device=self.device, dtype=self.dtype)
        self.optimizer_Adam = torch.optim.Adam(self.u_model.parameters(), lr=self.lr)
        open(history_path, "w").close()  # fresh history file for this run

        n_data = self.x.shape[0]
        n_pi = self.x_pi.shape[0]
        for i_iter in range(n_iters):
            start_time = time.time()
            i = np.random.choice(n_data, batch_data, replace=False)
            i_pi = np.random.choice(n_pi, batch_pi, replace=False)
            tmpx, tmpy = self.x[i], self.y[i]
            tmpu, tmpv, tmpp = self.u[i], self.v[i], self.p[i]
            tmpuu, tmpuv, tmpvv = self.uu[i], self.uv[i], self.vv[i]
            tmpxpi, tmpypi = self.x_pi[i_pi], self.y_pi[i_pi]

            loss_value, grads = self.grad(tmpx, tmpy, tmpu, tmpv, tmpp, tmpuu, tmpuv, tmpvv, tmpxpi, tmpypi)
            for param, g in zip(self.u_model.parameters(), grads):
                param.grad = g
            self.optimizer_Adam.step()
            elapsed = time.time() - start_time

            if i_iter % print_interval == 0:
                loss_uvbc, loss_uuuvvvbc, loss_divuv, loss_uuuvvv = self.monitor(
                    tmpx, tmpy, tmpu, tmpv, tmpp, tmpuu, tmpuv, tmpvv, tmpxpi, tmpypi)
                print('I iter: %d, Loss: %.5e, uvbc: %.5e, uuuvvvbc: %.5e, divuv: %.5e, uuuvvv: %.5e, Time: %.2f' %
                      (i_iter, loss_value, loss_uvbc, loss_uuuvvvbc, loss_divuv, loss_uuuvvv, elapsed))
                with open(history_path, "a") as myfile:
                    myfile.write('I iter: %d, Loss: %.5e, uvbc: %.5e, uuuvvvbc: %.5e, divuv: %.5e, uuuvvv: %.5e, Time: %.2f, \n' %
                                  (i_iter, loss_value, loss_uvbc, loss_uuuvvvbc, loss_divuv, loss_uuuvvv, elapsed))

    def train_lbfgs(self, n_lbfgs, history_path, lbfgs_options=None, batch_mode="full",
                     batch_data=None, batch_pi=None):
        lbfgs_options = lbfgs_options or {}
        params = list(self.u_model.parameters())
        open(history_path, "w").close()  # fresh history file for this run

        ftol = lbfgs_options.get("ftol", "eps")
        tolerance_change = np.finfo(float).eps if ftol == "eps" else ftol

        for i in range(n_lbfgs):
            if batch_mode == "batch":
                idx = np.random.choice(self.x.shape[0], batch_data, replace=False)
                idx_pi = np.random.choice(self.x_pi.shape[0], batch_pi, replace=False)
                x, y = self.x[idx], self.y[idx]
                u, v, p = self.u[idx], self.v[idx], self.p[idx]
                uu, uv, vv = self.uu[idx], self.uv[idx], self.vv[idx]
                x_pi, y_pi = self.x_pi[idx_pi], self.y_pi[idx_pi]
            else:
                x, y, u, v, p, uu, uv, vv, x_pi, y_pi = (
                    self.x, self.y, self.u, self.v, self.p, self.uu, self.uv, self.vv, self.x_pi, self.y_pi)

            optimizer = torch.optim.LBFGS(
                params, lr=1.0,
                max_iter=lbfgs_options.get("maxiter", 200),
                max_eval=lbfgs_options.get("max_eval", 2000),
                tolerance_grad=lbfgs_options.get("tolerance_grad", 1e-8),
                tolerance_change=tolerance_change,
                history_size=lbfgs_options.get("maxcor", 100),
                line_search_fn=lbfgs_options.get("line_search_fn", 'strong_wolfe'),
            )

            def closure():
                optimizer.zero_grad()
                group_uvbc, group_stressbc, group_momentum, group_stresstransport = self._loss_groups(
                    x, y, u, v, p, uu, uv, vv, x_pi, y_pi)
                loss = group_uvbc + group_stressbc + group_momentum + group_stresstransport
                loss.backward()
                return loss

            start_time = time.time()
            loss = optimizer.step(closure)
            elapsed = time.time() - start_time

            opt_state = optimizer.state[params[0]]
            n_iter_actual = opt_state.get("n_iter", -1)
            func_evals = opt_state.get("func_evals", -1)

            loss_uvbc, loss_uuuvvvbc, loss_divuv, loss_uuuvvv = self.monitor(
                x, y, u, v, p, uu, uv, vv, x_pi, y_pi)
            with open(history_path, "a") as myfile:
                myfile.write('I iter: %d, Loss: %.5e, uvbc: %.5e, uuuvvvbc: %.5e, divuv: %.5e, uuuvvv: %.5e, '
                              'n_iter: %d, func_evals: %d, Time: %.2f, \n' %
                              (i, loss.item(), loss_uvbc, loss_uuuvvvbc, loss_divuv, loss_uuuvvv,
                               n_iter_actual, func_evals, elapsed))

    def predict(self, x_star, y_star):
        x_star = torch.as_tensor(x_star, dtype=self.dtype, device=self.device)
        y_star = torch.as_tensor(y_star, dtype=self.dtype, device=self.device)
        if self.use_coord_scaling:
            x_star = x_star / self.Lxscale
            y_star = y_star / self.Lyscale
        with torch.no_grad():
            utau_star = self.u_model(torch.cat([x_star, y_star], dim=-1))
            u = utau_star[:, 0:1]
            v = utau_star[:, 1:2]
            p = utau_star[:, 2:3]

            if self.if_positivity[0] is False:
                tau1 = utau_star[:, 3:4]
                tau3 = utau_star[:, 5:6]
            else:
                if self.if_positivity[1] == 'softplus':
                    tau1 = F.softplus(utau_star[:, 3:4])
                    tau3 = F.softplus(utau_star[:, 5:6])
                elif self.if_positivity[1] == 'relu':
                    tau1 = F.relu(utau_star[:, 3:4])
                    tau3 = F.relu(utau_star[:, 5:6])
                else:
                    raise ValueError(f"unsupported if_positivity mode: {self.if_positivity[1]!r}")

            if self.if_realisability is False:
                tau2 = utau_star[:, 4:5]
            else:
                tau2 = torch.sqrt(tau1) * torch.sqrt(tau3) * torch.tanh(utau_star[:, 4:5])

        return (u.cpu().numpy(), v.cpu().numpy(), p.cpu().numpy(),
                tau1.cpu().numpy(), tau2.cpu().numpy(), tau3.cpu().numpy())

    def save(self, path):
        torch.save(self.u_model.state_dict(), path + 'PINNS_u.pt')

    def restore(self, path):
        self.u_model = FCNN(self.layers_u).to(device=self.device, dtype=self.dtype)
        state_dict = torch.load(path + 'PINNS_u.pt', map_location=self.device)
        self.u_model.load_state_dict(state_dict)
        self.u_model.eval()
