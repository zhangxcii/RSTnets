import numpy as np
import time
import tensorflow as tf
import deepxde as dde
from ANN import FCNN

class RSTnet:
    def __init__(self, invRe, train_dd, train_pi, layers_u, lr, if_positivity, if_realisability):
        self.invRe =  invRe
        x, y, u, v, p, uu, uv, vv = train_dd[:,0:1], train_dd[:,1:2], train_dd[:,2:3], train_dd[:,3:4], train_dd[:,4:5], train_dd[:,5:6], train_dd[:,6:7], train_dd[:,7:8]
        x_pi, y_pi = train_pi[:,0:1], train_pi[:,1:2]

        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.p = p
        self.uu = uu
        self.uv = uv
        self.vv = vv
        self.x_pi = x_pi
        self.y_pi = y_pi

        self.layers_u = layers_u
        self.lr = lr
        self.if_positivity = if_positivity
        self.if_realisability = if_realisability
        
    def net_RANS(self, x, y):
        utauf = self.u_model(tf.concat([x, y], -1))
        u = utauf[:,0:1]
        v = utauf[:,1:2]
        p = utauf[:,2:3] 
        if self.if_positivity[0] == False:
            tau1 = utauf[:,3:4]
            tau3 = utauf[:,5:6]
        elif self.if_positivity[0] == True:
            if self.if_positivity[1] == 'softplus':
                tau1 = tf.nn.softplus(utauf[:,3:4])
                tau3 = tf.nn.softplus(utauf[:,5:6])
            elif self.if_positivity[1] == 'relu':
                tau1 = tf.nn.relu(utauf[:,3:4])
                tau3 = tf.nn.relu(utauf[:,5:6])
            else:
                print('Options not supported yet (J ZHANG)')

        if self.if_realisability == False:
            tau2 = utauf[:,4:5]
        else:
            tau2 = tf.math.sqrt(tau1)*tf.math.sqrt(tau3)*tf.nn.tanh(utauf[:,4:5])

        ftau1 = utauf[:,6:7]
        ftau2 = utauf[:,7:8]
        ftau3 = utauf[:,8:9]

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
        f_u = (u*u_x + v*u_y ) + p_x - self.invRe*(u_xx + u_yy ) + tau1_x + tau2_y
        f_v = (u*v_x + v*v_y ) + p_y - self.invRe*(v_xx + v_yy ) + tau2_x + tau3_y
        f_tau1 = (u*tau1_x + v*tau1_y)  + ftau1
        f_tau2 = (u*tau2_x + v*tau2_y)  + ftau2
        f_tau3 = (u*tau3_x + v*tau3_y)  + ftau3

        return u, v, p, tau1, tau2, tau3, f_div, f_u, f_v, f_tau1, f_tau2, f_tau3

    @tf.function
    def grad(self, x, y, u, v, p, uu, uv, vv, x_pi, y_pi):
        with tf.GradientTape() as tape:
            self.u_pred, self.v_pred, self.p_pred, self.uu_pred, self.uv_pred, self.vv_pred,  _, _, _, _, _, _  = self.net_RANS(x, y)
            _, _, _, _, _, _, self.f_div ,self.f_u_pred, self.f_v_pred, self.f_tau1_pred, self.f_tau2_pred, self.f_tau3_pred  = self.net_RANS(x_pi, y_pi)

            self.loss =  tf.reduce_mean(tf.square(u - self.u_pred)) +\
            tf.reduce_mean(tf.square(v - self.v_pred)) +\
            tf.reduce_mean(tf.square(uu - self.uu_pred)) +\
            tf.reduce_mean(tf.square(uv - self.uv_pred)) +\
            tf.reduce_mean(tf.square(vv - self.vv_pred)) +\
            tf.reduce_mean(tf.square(self.f_div))+ \
            tf.reduce_mean(tf.square(self.f_u_pred))+\
            tf.reduce_mean(tf.square(self.f_v_pred))+\
            tf.reduce_mean(tf.square(self.f_tau1_pred))+\
            tf.reduce_mean(tf.square(self.f_tau2_pred))+\
            tf.reduce_mean(tf.square(self.f_tau3_pred)) 
            self.grads = tape.gradient(self.loss, self.u_model.trainable_variables)

        return self.loss, self.grads

    @tf.function
    def monitor(self, x, y, u, v, p, uu, uv, vv, x_pi, y_pi):
        with tf.GradientTape() as tape:
            self.u_pred, self.v_pred, self.p_pred, self.uu_pred, self.uv_pred, self.vv_pred,  _, _, _, _, _, _  = self.net_RANS(x, y)
            _, _, _, _, _, _, self.f_div ,self.f_u_pred, self.f_v_pred, self.f_tau1_pred, self.f_tau2_pred, self.f_tau3_pred  = self.net_RANS(x_pi, y_pi)

        return tf.reduce_mean(tf.square(u - self.u_pred))+tf.reduce_mean(tf.square(v - self.v_pred)),  tf.reduce_mean(tf.square(uu - self.uu_pred))+tf.reduce_mean(tf.square(uv - self.uv_pred))+tf.reduce_mean(tf.square(vv - self.vv_pred)),  tf.reduce_mean(tf.square(self.f_div))+tf.reduce_mean(tf.square(self.f_u_pred))+tf.reduce_mean(tf.square(self.f_v_pred)),  tf.reduce_mean(tf.square(self.f_tau1_pred))+tf.reduce_mean(tf.square(self.f_tau2_pred))+tf.reduce_mean(tf.square(self.f_tau3_pred)) 

    def train(self, n_iters, batch_data, batch_pi, print_interval = 100):
        self.u_model =  FCNN(self.layers_u)
        self.optimizer_Adam = tf.keras.optimizers.Adam(learning_rate=self.lr)
        for i_iter in range(n_iters) :
            start_time = time.time()
            i =  np.random.choice(self.x.shape[0],batch_data, replace=False) 
            i_pi = np.random.choice(self.x_pi.shape[0], batch_pi, replace=False)
            tmpx = tf.cast(self.x[i],tf.float32)
            tmpy = tf.cast(self.y[i],tf.float32)
            tmpu = tf.cast(self.u[i],tf.float32)
            tmpv = tf.cast(self.v[i],tf.float32)
            tmpp = tf.cast(self.p[i],tf.float32)
            tmpuu = tf.cast(self.uu[i],tf.float32)
            tmpuv = tf.cast(self.uv[i],tf.float32)
            tmpvv = tf.cast(self.vv[i],tf.float32)
            tmpxpi = tf.cast(  self.x_pi[i_pi],tf.float32)
            tmpypi = tf.cast(  self.y_pi[i_pi],tf.float32)

            loss_value, grads = self.grad(tmpx, tmpy, tmpu, tmpv, tmpp, tmpuu, tmpuv, tmpvv, tmpxpi, tmpypi)
            self.optimizer_Adam.apply_gradients(zip(grads, self.u_model.trainable_variables))
            elapsed = time.time() - start_time

            if i_iter%print_interval == 0:
                loss_uvbc, loss_uuuvvvbc, loss_divuv, loss_uuuvvv = self.monitor(tmpx, tmpy, tmpu, tmpv, tmpp, tmpuu, tmpuv, tmpvv, tmpxpi, tmpypi)
                print('I iter: %d, Loss: %.5e, uvbc: %.5e, uuuvvvbc: %.5e, divuv: %.5e, uuuvvv: %.5e, Time: %.2f' % 
                  (i_iter, loss_value, loss_uvbc, loss_uuuvvvbc, loss_divuv, loss_uuuvvv, elapsed))
                with open("train_hist.txt", "a") as myfile:
                    myfile.write('I iter: %d, Loss: %.5e, uvbc: %.5e, uuuvvvbc: %.5e, divuv: %.5e, uuuvvv: %.5e, Time: %.2f, \n' % (i_iter, loss_value, loss_uvbc, loss_uuuvvvbc, loss_divuv, loss_uuuvvv, elapsed))

    def train_lbfgs(self, n_lbfgs):
        trainable_variables = self.u_model.trainable_variables 
        def build_loss():
            with tf.GradientTape() as tape:
                self.u_pred, self.v_pred, self.p_pred, self.uu_pred, self.uv_pred, self.vv_pred,  _, _, _, _, _, _  = self.net_RANS(tf.cast(self.x,tf.float32), tf.cast(self.y,tf.float32))
                _, _, _, _, _, _, self.f_div ,self.f_u_pred, self.f_v_pred, self.f_tau1_pred, self.f_tau2_pred, self.f_tau3_pred  = self.net_RANS(tf.cast(  self.x_pi,tf.float32), tf.cast(  self.y_pi,tf.float32))

                self.loss =  tf.reduce_mean(tf.square(tf.cast(self.u,tf.float32) - self.u_pred)) +\
                tf.reduce_mean(tf.square(tf.cast(self.v,tf.float32) - self.v_pred)) +\
                tf.reduce_mean(tf.square(tf.cast(self.uu,tf.float32) - self.uu_pred)) +\
                tf.reduce_mean(tf.square(tf.cast(self.uv,tf.float32) - self.uv_pred)) +\
                tf.reduce_mean(tf.square(tf.cast(self.vv,tf.float32) - self.vv_pred)) +\
                tf.reduce_mean(tf.square(self.f_div))+ \
                tf.reduce_mean(tf.square(self.f_u_pred))+\
                tf.reduce_mean(tf.square(self.f_v_pred))+\
                tf.reduce_mean(tf.square(self.f_tau1_pred))+\
                tf.reduce_mean(tf.square(self.f_tau2_pred))+\
                tf.reduce_mean(tf.square(self.f_tau3_pred)) 
            return self.loss 

        for i in range(n_lbfgs):
            start_time = time.time()
            dde.optimizers.config.set_LBFGS_options(maxcor=100,ftol=1.0 * np.finfo(float).eps,maxiter=200,maxls=50)
            results = dde.optimizers.tensorflow.tfp_optimizer.lbfgs_minimize(trainable_variables, build_loss)
            elapsed = time.time() - start_time

            loss_uvbc, loss_uuuvvvbc, loss_divuv, loss_uuuvvv = self.monitor(tf.cast(self.x,tf.float32), tf.cast(self.y,tf.float32), tf.cast(self.u,tf.float32), tf.cast(self.v,tf.float32), tf.cast(self.p,tf.float32), tf.cast(self.uu,tf.float32), tf.cast(self.uv,tf.float32), tf.cast(self.vv,tf.float32), tf.cast(self.x_pi,tf.float32), tf.cast(self.y_pi,tf.float32))
            with open("train_hist_lbfgs.txt", "a") as myfile:
                myfile.write('I iter: %d, Loss: %.5e, uvbc: %.5e, uuuvvvbc: %.5e, divuv: %.5e, uuuvvv: %.5e, Time: %.2f, \n' % (i, results.objective_value, loss_uvbc, loss_uuuvvvbc, loss_divuv, loss_uuuvvv, elapsed))            
   
    def predict(self, x_star, y_star):
        x_star = tf.cast( x_star, tf.float32)
        y_star = tf.cast( y_star, tf.float32)
        utau_star  = self.u_model(tf.concat([x_star,y_star], -1))
        u = utau_star[:,0:1]
        v = utau_star[:,1:2]
        p = utau_star[:,2:3]  

        if self.if_positivity[0] == False:
            tau1 = utau_star[:,3:4]
            tau3 = utau_star[:,5:6]
        elif self.if_positivity[0] == True:
            if self.if_positivity[1] == 'softplus':
                tau1 = tf.nn.softplus(utau_star[:,3:4])
                tau3 = tf.nn.softplus(utau_star[:,5:6])
            elif self.if_positivity[1] == 'relu':
                tau1 = tf.nn.relu(utau_star[:,3:4])
                tau3 = tf.nn.relu(utau_star[:,5:6])
            else:
                print('Options not supported yet (J ZHANG)')

        if self.if_realisability == False:
            tau2 = utau_star[:,4:5]
        else:
            tau2 = tf.math.sqrt(tau1)*tf.math.sqrt(tau3)*tf.nn.tanh(utau_star[:,4:5])    

        return u.numpy(), v.numpy(), p.numpy(), tau1.numpy(),tau2.numpy(), tau3.numpy()
        
    def save(self,path):
        self.u_model.save_weights(path +'PINNS_u')

    def restore(self,path):
        self.u_model =  FCNN(self.layers_u)
        self.u_model.load_weights(path+'PINNS_u')




















