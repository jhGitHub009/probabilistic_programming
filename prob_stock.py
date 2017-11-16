import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pyro
from pyro.distributions import Normal
from pyro.infer import SVI
from pyro.optim import Adam

class RegressionModel(nn.Module):
    def __init__(self, p):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)   # input = p weight = 1 bias = none

    def forward(self, x):
        return self.linear(x)

class test_linear:
    def __init__(self,input_size, output_size, name="main"):
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self.regression_model = RegressionModel(self.input_size)
    def model(self, data):
        # Create unit normal priors over the parameters
        x_data = data[:, :-1]
        y_data = data[:, -1]
        # mu, sigma = Variable(torch.zeros(self.input_size, 1)), Variable(10 * torch.ones(self.input_size, 1))
        mu, sigma = Variable(torch.zeros(1,self.input_size)), Variable(10 * torch.ones(1,self.input_size))
        bias_mu, bias_sigma = Variable(torch.zeros(1)), Variable(10 * torch.ones(1))
        w_prior, b_prior = Normal(mu, sigma), Normal(bias_mu, bias_sigma)  # mean = mu, std = 1 평균이 0이고 std가 1인 정규 분포
        priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.regression_model, priors)
        # sample a regressor (which also samples w and b)
        self.lifted_reg_model = lifted_module()
        # run the regressor forward conditioned on data
        # test = self.lifted_reg_model(x_data)
        prediction_mean = self.lifted_reg_model(x_data).squeeze()  # 100,1을 [100]으로 만듬 remove 1 dimension.
        # condition on the observed data
        pyro.observe("obs", Normal(prediction_mean, Variable(0.1 * torch.ones(data.size(0)))),y_data.squeeze())  # return sample
        # name = "obs", fn = distribution class or function, obs - observe d datum

    def guide(self, data):
        # define our variational parameters
        softplus = torch.nn.Softplus()
        w_mu = Variable(torch.randn(1, self.input_size), requires_grad=True)
        w_log_sig = Variable(-3.0 * torch.ones(1, self.input_size) + 0.05 * torch.randn(self.input_size), requires_grad=True)
        b_mu = Variable(torch.randn(1), requires_grad=True)
        b_log_sig = Variable(-3.0 * torch.ones(1) + 0.05 * torch.randn(1), requires_grad=True)
        # register learnable params in the param store
        mw_param = pyro.param("guide_mean_weight",w_mu)
        sw_param = softplus(pyro.param("guide_log_sigma_weight",w_log_sig))
        mb_param = pyro.param("guide_mean_bias", b_mu)
        sb_param = softplus(pyro.param("guide_log_sigma_bias", b_log_sig))
        # guide distributions for w and b
        w_dist, b_dist = Normal(mw_param, sw_param), Normal(mb_param, sb_param)
        # w_dist, b_dist = Normal(mw_param), Normal(mb_param)
        dists = {'linear.weight': w_dist, 'linear.bias': b_dist}
        # overload the parameters in the module with random samples
        # from the guide distributions
        lifted_module = pyro.random_module("module", self.regression_model, dists)
        # sample a regressor (which also samples w and b)
        return lifted_module()

    def predict(self, data):
        # x = np.reshape(state, [1, self.input_size])
        y_pred = self.lifted_reg_model(data)
        # return self.session.run(self._Qpred, feed_dict={self._X: x})
        return y_pred
    def update(self, svi, x_stack, y_stack):
        data = torch.cat((x_stack, y_stack), 1)
        loss = svi.step(data)
        return loss
        # return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})

def build_linear_dataset(N, noise_std=0.1):
    X = np.linspace(-6, 6, num=N)   # -6 ~ 6 까지 내 일정한 상수 N개를 만든다.
    if __name__ == '__main__':
        y = 3 * X + 1 + np.random.normal(0, noise_std, size=N)  # random한 변수를 std=0.1 로 N개 만들고 y=3x+1 에 노이즈로 더한다
    X, y = X.reshape((N, 1)), y.reshape((N, 1))     # 변수를 N행 1렬로 변환한다.
    if __name__ == '__main__':
        X, y = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))     # Variable로 변환
    return torch.cat((X, y), 1)         #x,y 를 합치는데 행을 기준으로 합친다. N행 2열이됨.

import pandas as pd
def load_data():
    df = pd.read_csv('./000020.csv',index_col=0)
    df['Open'] = df['Open'] / df['Open'].max()
    df['High'] = df['High'] / df['High'].max()
    df['Low'] = df['Low'] / df['Low'].max()
    df['Volume'] = df['Volume'] / df['Volume'].max()
    df['Close'] = df['Close'] / df['Close'].max()
    x_data = df[['Open', 'High', 'Low', 'Volume']].as_matrix()
    y_data = df['Close'].as_matrix()
    X, y = x_data.reshape((len(x_data), 4)), y_data.reshape((len(x_data), 1))
    X, y = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))
    return torch.cat((X, y), 1)         #x,y 를 합치는데 행을 기준으로 합친다. N행 2열이됨.

if __name__ == '__main__':
    m = nn.Linear(20, 1)
    input = Variable(torch.randn(128, 20))
    output = m(input)
    print(output.size())

    pyro.clear_param_store()
    N = 100  # size of toy data
    p = 1  # number of features
    num_iterations = 500
    data = load_data()
    test_reg = test_linear(input_size=4, output_size=1)
    optim = Adam({"lr": 0.01})
    svi = SVI(test_reg.model, test_reg.guide, optim, loss="ELBO")
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(data[0:len(data)-100])
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / (len(data)-100)))

    x_data = data[len(data) - 50:len(data) - 1, :-1]
    y_pred = test_reg.predict(x_data)
    print(y_pred)
    print('debug')
    for name in pyro.get_param_store().get_all_param_names():
        print('[%s] : '%(name)+str(pyro.param(name).data.numpy()))