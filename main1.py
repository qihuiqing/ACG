# coding: utf-8
'''
Created on 2021年1月21日
@author: Chyi
'''
#import the all modules
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.optim.optimizer import required

from ACG_Visualization import kde,init_weights
from ACG_Optimizer import *

#hyper-parameters
OUTPUT="ACG_OURS_result"
n_iter = 10001
n_save = 1000
z_dim = 64
batch_size = 300
bbox = [-2, 2, -2, 2]
flag_save = True
flag_show = True
depth = 6

#define the network
class MLP(nn.Module):
    def __init__(self, depth, latent_dim, output_dim):
        super(MLP, self).__init__()
        self.depth = depth
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = 256 # the width of the networks
        self.layers = []
        self.layers.append(nn.Linear(self.latent_dim, self.hidden_dim))
        self.layers.append(nn.ReLU(True))
        
        for _ in range(self.depth-2):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.ReLU(True))
        
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        self.layers = nn.Sequential(*self.layers)
         
    def forward(self, input):
        return self.layers(input)

#usually functions

def flatten(tensor_seq):
    tens_flatten = tensor_seq[0].view(-1)
    for t in tensor_seq[1:]:
        tens_flatten = torch.cat((tens_flatten, t.view(-1)))
    return tens_flatten

def compute_coeff(alpha_vecs):#计算预测系数的
    vecs = [vec.view(-1, 1) for vec in alpha_vecs[:-1]]
    A = torch.cat(vecs, 1)
    y = alpha_vecs[-1].view(-1, 1)
    return torch.pinverse(A) @ y

def diff_vect(vars_list):
    diff_vars=[]
    for i in range(len(vars_list)-1):
        diff_vars.append(vars_list[i+1]-vars_list[i])
    return diff_vars

def LP_fun(grads_list, diff_vars, c, lr, lrt, s=200):#计算新的合成梯度
    grads = [vec.view(-1, 1) for vec in grads_list]
    v_k = [vec.view(-1, 1) for vec in diff_vars]
    v_k = torch.cat(v_k,1)
    grads = torch.cat(grads,1)
    C = np.diag((1.0,1.0,1.0),-1)
    b=np.array(c.tolist())
    C[:,-1]=b[:,0]
    pc = C
    #有限步预测
    for _ in range(s+1):
        pc = pc@C
    s =(C-pc) @ np.linalg.pinv(np.eye(4)-C)
    s = torch.cuda.FloatTensor(s)
    #无限步预测
#     s = C @ np.linalg.pinv(np.eye(4)-C)
#     s = torch.cuda.FloatTensor(s)
    m = v_k.shape[1]
    e = v_k[:,1:m] @ s[:,-1] #新梯度
    #计算向心方向分量
    A1 = grads[:,-1]-grads[:,-2]
    grads_e = grads[:,-2] / torch.norm(grads[:,-2],2)
    num = torch.dot(grads_e,A1)
    AC = A1-num*grads_e
    e = e+ lr*grads[:,-1]+lrt*AC
#     e = e+lrt*AC
    return e

def x_real_builder(batch_size):
    sigma = 0.1
    grid = np.array([
        [1.00, 1.00],
        [1.00, -1.00],
        [-1.00, 1.00],
        [-1.00, -1.00],
        [0.0,  0.0],
    ]
    )
    temp = np.tile(grid, (batch_size // 5 + 1, 1))
    mus = temp[0:batch_size, :]
    arr = mus + sigma * np.random.randn(batch_size, 2) * .2
    return arr


def train(learning_rate,n_iter=20001,n_save=1000,depth=4,z_dim=64,batch_size=300):
    ztest = [torch.randn(batch_size, z_dim).cuda() for i in range(10)]
    disc_loss_hist = []
    gen_loss_hist = []
    x_real = torch.cuda.FloatTensor(x_real_builder(batch_size))
    x_dim = x_real.shape[1]
    generator = MLP(depth, z_dim, x_dim).cuda()
    discriminator = MLP(depth, x_dim, 1).cuda()
    generator.apply(init_weights)
    discriminator.apply(init_weights)
        
    loss = nn.BCEWithLogitsLoss()
    params = [{'params':generator.parameters()}, \
            {'params':discriminator.parameters()}]
    lr = learning_rate * 1e-4
    optimizer = SymplecticOptimizer(params, lr=lr, alpha=0.95)
#     optimizer = myRMSProp(params, lr=lr, alpha=0.97)
    optimizer1 = myoptimizer(params, lr=lr, alpha=0.9)
    grads_list = []
    vars_list = []
    for i in tqdm(range(1,n_iter)):
        discriminator.zero_grad()
        generator.zero_grad() 
        z = torch.randn([batch_size, z_dim]).cuda()
        x_fake = generator(z) 
        disc_out_real = discriminator(x_real)
        disc_out_fake = discriminator(x_fake)
        
        disc_loss_real = loss(disc_out_real, torch.ones_like(disc_out_real).cuda())
        disc_loss_fake = loss(disc_out_fake, torch.zeros_like(disc_out_fake).cuda())
        disc_loss = torch.add(disc_loss_real, disc_loss_fake)
        gen_loss = loss(disc_out_fake, torch.ones_like(disc_out_fake).cuda())

        if i % 7 !=0:
            grads, Vars = optimizer.step([gen_loss, disc_loss])
            xs = flatten(Vars)
            grad = flatten(grads)
            grads_list.append(grad.detach())
            vars_list.append(xs.detach())
        else:
            diff_vars = diff_vect(vars_list)
            coef = compute_coeff(diff_vars)
            e = LP_fun(grads_list, diff_vars, coef, lr, lrt=0.001)
            optimizer1.step(e)
            grads_list = []
            vars_list = []

        disc_loss_hist += [disc_loss.detach()]
        gen_loss_hist += [gen_loss.detach()]
        if (i % n_save == 0):
            print('i = %d, discriminant loss = %.4f, generator loss =%.4f' %
                (i, disc_loss.detach(), gen_loss.detach()))
            x_out = np.concatenate([generator(zt).cpu().detach().numpy() for zt in ztest], axis=0)
            ax=kde(x_out[:, 0], x_out[:, 1], bbox=bbox)        
            plt.savefig('./ACG_OURS_result/{}_{:.2f}.png'.format(i, learning_rate))
    plt.figure()
    plt.plot(gen_loss_hist, label='generator')
    plt.plot(disc_loss_hist, label='discrimitor')
    plt.legend()
    plt.savefig('./ACG_OURS_result/result_{:.2f}.png'.format(learning_rate))
    
if __name__=="__main__":
    if not os.path.isdir(OUTPUT):
        os.mkdir(OUTPUT)
    learning_rates = np.arange(2.5, 7.5, 0.2)
    train(5.0, n_iter, n_save, depth, z_dim, batch_size)