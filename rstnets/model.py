import time

import numpy as np
import tensorflow as tf
import deepxde as dde

from .config import CaseConfig
from .networks import FCNN


class RSTnet:
    """Unified RSTnet: solves the unclosed RANS + Reynolds-stress-transport equations
    via a physics-informed neural network.
    """

    def __init__(self, case_config: CaseConfig, train_dd, train_pi, Lxscale=None, Lyscale=None, precision=None):
        self.invRe = case_config.invRe
        self.layers_u = case_config.network.layers
        self.lr = case_config.network.learning_rate
        self.if_positivity = case_config.if_positivity
        self.if_realisability = case_config.if_realisability
        self.use_coord_scaling = case_config.use_coord_scaling
        self.supervise_pressure = case_config.supervise_pressure

        precision = precision or case_config.precision
        self.dtype = tf.float64 if precision == "float64" else tf.float32

        if self.use_coord_scaling and (Lxscale is None or Lyscale is None):
            raise ValueError(f"case {case_config.name!r} requires coordinate scaling but "
                              "Lxscale/Lyscale were not provided")
        self.Lxscale = Lxscale
        self.Lyscale = Lyscale

        x, y = train_dd[:, 0:1], train_dd[:, 1:2]
        u, v, p, uu, uv, vv = (train_dd[:, i:i + 1] for i in range(2, 8))
        x_pi, y_pi = train_pi[:, 0:1], train_pi[:, 1:2]

        self.x, self.y = x, y
        self.u, self.v, self.p = u, v, p
        self.uu, self.uv, self.vv = uu, uv, vv
        self.x_pi, self.y_pi = x_pi, y_pi

    def net_RANS(self, x, y):
        if self.use_coord_scaling:
            x_s = x / self.Lxscale
            y_s = y / self.Lyscale
        else:
            x_s, y_s = x, y

        utauf = self.u_model(tf.concat([x_s, y_s], -1))
        u = utauf[:, 0:1]
        v = utauf[:, 1:2]
        p = utauf[:, 2:3]
        if self.if_positivity[0] is False:
            tau1 = utauf[:, 3:4]
            tau3 = utauf[:, 5:6]
        else:
            if self.if_positivity[1] == 'softplus':
                tau1 = tf.nn.softplus(utauf[:, 3:4])
                tau3 = tf.nn.softplus(utauf[:, 5:6])
            elif self.if_positivity[1] == 'relu':
                tau1 = tf.nn.relu(utauf[:, 3:4])
                tau3 = tf.nn.relu(utauf[:, 5:6])
            else:
                raise ValueError(f"unsupported if_positivity mode: {self.if_positivity[1]!r}")

        if self.if_realisability is False:
            tau2 = utauf[:, 4:5]
        else:
            tau2 = tf.math.sqrt(tau1) * tf.math.sqrt(tau3) * tf.nn.tanh(utauf[:, 4:5])

        ftau1 = utauf[:, 6:7]
        ftau2 = utauf[:, 7:8]
        ftau3 = utauf[:, 8:9]

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        tau1_x = tf.gradients(tau1, x)[0]
        tau1_y = tf.gradients(tau1, y)[0]
        tau2_x = tf.gradients(tau2, x)[0]
        tau2_y = tf.gradients(tau2, y)[0]
        tau3_x = tf.gradients(tau3, x)[0]
        tau3_y = tf.gradients(tau3, y)[0]

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

        group_uvbc = tf.reduce_mean(tf.square(u - u_pred)) + tf.reduce_mean(tf.square(v - v_pred))
        if self.supervise_pressure:
            group_uvbc = group_uvbc + tf.reduce_mean(tf.square(p - p_pred))
        group_stressbc = (tf.reduce_mean(tf.square(uu - uu_pred)) +
                           tf.reduce_mean(tf.square(uv - uv_pred)) +
                           tf.reduce_mean(tf.square(vv - vv_pred)))
        group_momentum = (tf.reduce_mean(tf.square(f_div)) +
                           tf.reduce_mean(tf.square(f_u_pred)) +
                           tf.reduce_mean(tf.square(f_v_pred)))
        group_stresstransport = (tf.reduce_mean(tf.square(f_tau1_pred)) +
                                  tf.reduce_mean(tf.square(f_tau2_pred)) +
                                  tf.reduce_mean(tf.square(f_tau3_pred)))
        return group_uvbc, group_stressbc, group_momentum, group_stresstransport

    @tf.function(jit_compile=True)
    def grad(self, x, y, u, v, p, uu, uv, vv, x_pi, y_pi):
        with tf.GradientTape() as tape:
            group_uvbc, group_stressbc, group_momentum, group_stresstransport = self._loss_groups(
                x, y, u, v, p, uu, uv, vv, x_pi, y_pi)
            loss = group_uvbc + group_stressbc + group_momentum + group_stresstransport
            grads = tape.gradient(loss, self.u_model.trainable_variables)
        return loss, grads

    @tf.function
    def monitor(self, x, y, u, v, p, uu, uv, vv, x_pi, y_pi):
        return self._loss_groups(x, y, u, v, p, uu, uv, vv, x_pi, y_pi)

    def _cast(self, arr):
        return tf.cast(arr, self.dtype)

    def train(self, n_iters, batch_data, batch_pi, history_path, print_interval=100):
        self.u_model = FCNN(self.layers_u)
        self.optimizer_Adam = tf.keras.optimizers.Adam(learning_rate=self.lr)
        open(history_path, "w").close()  # fresh history file for this run

        for i_iter in range(n_iters):
            start_time = time.time()
            i = np.random.choice(self.x.shape[0], batch_data, replace=False)
            i_pi = np.random.choice(self.x_pi.shape[0], batch_pi, replace=False)
            tmpx, tmpy = self._cast(self.x[i]), self._cast(self.y[i])
            tmpu, tmpv, tmpp = self._cast(self.u[i]), self._cast(self.v[i]), self._cast(self.p[i])
            tmpuu, tmpuv, tmpvv = self._cast(self.uu[i]), self._cast(self.uv[i]), self._cast(self.vv[i])
            tmpxpi, tmpypi = self._cast(self.x_pi[i_pi]), self._cast(self.y_pi[i_pi])

            loss_value, grads = self.grad(tmpx, tmpy, tmpu, tmpv, tmpp, tmpuu, tmpuv, tmpvv, tmpxpi, tmpypi)
            self.optimizer_Adam.apply_gradients(zip(grads, self.u_model.trainable_variables))
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
        trainable_variables = self.u_model.trainable_variables
        open(history_path, "w").close()  # fresh history file for this run

        ftol = lbfgs_options.get("ftol", "eps")
        ftol = np.finfo(float).eps if ftol == "eps" else ftol
        gtol = lbfgs_options.get("gtol", 1e-8)
        maxcor = lbfgs_options.get("maxcor", 100)
        maxiter = lbfgs_options.get("maxiter", 200)
        maxls = lbfgs_options.get("maxls", 50)
        dde.optimizers.config.set_LBFGS_options(
            maxcor=maxcor, ftol=ftol, gtol=gtol, maxiter=maxiter, maxls=maxls)

        tf32 = lbfgs_options.get("tf32", None)
        if tf32 is not None:
            tf.config.experimental.enable_tensor_float_32_execution(tf32)

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

            xt, yt = self._cast(x), self._cast(y)
            ut, vt, pt = self._cast(u), self._cast(v), self._cast(p)
            uut, uvt, vvt = self._cast(uu), self._cast(uv), self._cast(vv)
            xpit, ypit = self._cast(x_pi), self._cast(y_pi)

            def build_loss():
                group_uvbc, group_stressbc, group_momentum, group_stresstransport = self._loss_groups(
                    xt, yt, ut, vt, pt, uut, uvt, vvt, xpit, ypit)
                return group_uvbc + group_stressbc + group_momentum + group_stresstransport

            start_time = time.time()
            results = dde.optimizers.tensorflow.tfp_optimizer.lbfgs_minimize(trainable_variables, build_loss)
            elapsed = time.time() - start_time

            loss_uvbc, loss_uuuvvvbc, loss_divuv, loss_uuuvvv = self.monitor(
                xt, yt, ut, vt, pt, uut, uvt, vvt, xpit, ypit)
            with open(history_path, "a") as myfile:
                myfile.write('I iter: %d, Loss: %.5e, uvbc: %.5e, uuuvvvbc: %.5e, divuv: %.5e, uuuvvv: %.5e, Time: %.2f, \n' %
                              (i, results.objective_value, loss_uvbc, loss_uuuvvvbc, loss_divuv, loss_uuuvvv, elapsed))

    def predict(self, x_star, y_star):
        x_star = self._cast(x_star)
        y_star = self._cast(y_star)
        if self.use_coord_scaling:
            x_star = x_star / self.Lxscale
            y_star = y_star / self.Lyscale
        utau_star = self.u_model(tf.concat([x_star, y_star], -1))
        u = utau_star[:, 0:1]
        v = utau_star[:, 1:2]
        p = utau_star[:, 2:3]

        if self.if_positivity[0] is False:
            tau1 = utau_star[:, 3:4]
            tau3 = utau_star[:, 5:6]
        else:
            if self.if_positivity[1] == 'softplus':
                tau1 = tf.nn.softplus(utau_star[:, 3:4])
                tau3 = tf.nn.softplus(utau_star[:, 5:6])
            elif self.if_positivity[1] == 'relu':
                tau1 = tf.nn.relu(utau_star[:, 3:4])
                tau3 = tf.nn.relu(utau_star[:, 5:6])
            else:
                raise ValueError(f"unsupported if_positivity mode: {self.if_positivity[1]!r}")

        if self.if_realisability is False:
            tau2 = utau_star[:, 4:5]
        else:
            tau2 = tf.math.sqrt(tau1) * tf.math.sqrt(tau3) * tf.nn.tanh(utau_star[:, 4:5])

        return u.numpy(), v.numpy(), p.numpy(), tau1.numpy(), tau2.numpy(), tau3.numpy()

    def save(self, path):
        self.u_model.save_weights(path + 'PINNS_u.weights.h5')

    def restore(self, path):
        self.u_model = FCNN(self.layers_u)
        self.u_model.load_weights(path + 'PINNS_u.weights.h5')
