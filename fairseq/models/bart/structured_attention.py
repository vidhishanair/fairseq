import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from .bilinear_matrix_attention import BilinearMatrixAttention


class StructuredAttention(nn.Module):
    def __init__(self, sent_hiddent_size, bidirectional, py_version, identity_init=False):
        super(StructuredAttention, self).__init__()
        self.bidirectional = bidirectional
        self.sem_dim_size = sent_hiddent_size//2
        self.str_dim_size = sent_hiddent_size - self.sem_dim_size
        self.pytorch_version = py_version
        print("Setting pytorch "+self.pytorch_version+" version for Structured Attention")

        self.tp_linear = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        self.tc_linear = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        self.fi_linear = nn.Linear(self.str_dim_size, 1, bias=False)

        # self.bilinear = nn.Bilinear(self.str_dim_size, self.str_dim_size, 1, bias=False)
        # torch.nn.init.xavier_uniform_(self.bilinear.weight)
        self.bilinear = BilinearMatrixAttention(self.str_dim_size, self.str_dim_size, False, 1)
        self.exparam = nn.Parameter(torch.Tensor(1,1,self.sem_dim_size))
        self.fzlinear = nn.Linear(3*self.sem_dim_size, self.sem_dim_size, bias=True)

        # initializations
        if identity_init == False:
            nn.init.xavier_uniform_(self.tp_linear.weight)
            nn.init.xavier_uniform_(self.tc_linear.weight)
            nn.init.xavier_uniform_(self.fi_linear.weight)
            nn.init.xavier_uniform_(self.exparam)
            nn.init.xavier_uniform_(self.fzlinear.weight)
        else:
            nn.init.eye_(self.tp_linear.weight)
            nn.init.eye_(self.tc_linear.weight)
            nn.init.eye_(self.fi_linear.weight)
            nn.init.ones_(self.exparam)
            nn.init.eye_(self.fzlinear.weight)
        nn.init.constant_(self.tp_linear.bias, 0)
        nn.init.constant_(self.tc_linear.bias, 0)
        nn.init.constant_(self.fzlinear.bias, 0)

    def forward(self, input): #batch*sent * token * hidden
        batch_size, token_size, dim_size = input.size()
        #print(input)
        if(self.bidirectional):
            input = input.view(batch_size, token_size, 2, dim_size//2)
            sem_v = torch.cat((input[:,:,0,:self.sem_dim_size//2],input[:,:,1,:self.sem_dim_size//2]),2)
            str_v = torch.cat((input[:,:,0,self.sem_dim_size//2:],input[:,:,1,self.sem_dim_size//2:]),2)
        else:
            sem_v = input[:,:,:self.sem_dim_size]
            str_v = input[:,:,self.sem_dim_size:]

        tp = F.tanh(self.tp_linear(str_v)) # b*s, token, h1
        tc = F.tanh(self.tc_linear(str_v)) # b*s, token, h1

        # tp = tp.unsqueeze(2).expand(tp.size(0), tp.size(1), tp.size(1), tp.size(2)).contiguous()
        # tc = tc.unsqueeze(2).expand(tc.size(0), tc.size(1), tc.size(1), tc.size(2)).contiguous()

        f_ij = F.tanh(self.bilinear(tp, tc).view(batch_size, token_size, token_size)) #.squeeze() # b*s, token , token
        f_i = torch.exp(F.tanh(self.fi_linear(str_v)).view(batch_size, token_size))  # b*s, token

        mask = f_ij.new_ones((f_ij.size(1), f_ij.size(1))) - f_ij.new_tensor(torch.eye(f_ij.size(1), f_ij.size(1)))
        mask = mask.unsqueeze(0).expand(f_ij.size(0), mask.size(0), mask.size(1)) #.to(self.device)

        A_ij = torch.exp(f_ij)*mask


        tmp = torch.sum(A_ij, dim=1)
        res = A_ij.new_zeros((batch_size, token_size, token_size)) #.to(self.device)
        #tmp = torch.stack([torch.diag(t) for t in tmp])
        res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)

        L_ij = -A_ij + res   #A_ij has 0s as diagonals

        L_ij_bar = L_ij.clone()
        #print(L_ij_bar.size(), f_i.size())
        L_ij_bar[:,0,:] = f_i

        #No batch inverse
        #LLinv = torch.stack([torch.inverse(li) for li in L_ij_bar])
        #print(f_ij)
        #print(f_i)
        #print(L_ij_bar)
        #print(np.linalg.det(L_ij_bar.data.cpu().numpy()))
        #exit()
        LLinv = None
        if self.pytorch_version == 'nightly' or self.pytorch_version=='1.3.0':
            LLinv = torch.inverse(L_ij_bar)
        else:
            LLinv = torch.stack([torch.inverse(li) for li in L_ij_bar])

        d0 = f_i * LLinv[:,:,0]

        LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)

        tmp1 = (A_ij.transpose(1,2) * LLinv_diag ).transpose(1,2)
        tmp2 = A_ij * LLinv.transpose(1,2)

        temp11 = A_ij.new_zeros((batch_size, token_size, 1))
        temp21 = A_ij.new_zeros((batch_size, 1, token_size))

        temp12 = A_ij.new_ones((batch_size, token_size, token_size-1))
        temp22 = A_ij.new_ones((batch_size, token_size-1, token_size))

        mask1 = torch.cat([temp11,temp12],2) #.to(self.device)
        mask2 = torch.cat([temp21,temp22],1) #.to(self.device)

        dx = mask1 * tmp1 - mask2 * tmp2

        #print(dx)
        d = torch.cat([d0.unsqueeze(1), dx], dim = 1)
        df = d.transpose(1,2)

        ssr = torch.cat([self.exparam.repeat(batch_size,1,1), sem_v], 1)
        pinp = torch.bmm(df, ssr)

        cinp = torch.bmm(dx, sem_v)

        finp = torch.cat([sem_v, pinp, cinp],dim = 2)

        output = F.relu(self.fzlinear(finp))
        #output = self.fzlinear(finp)
        #output = F.tanh(self.fzlinear(finp))

        self.output = output

        return output, df

def b_inv(b_mat, device):
    eye = torch.rand(b_mat.size(0), b_mat.size(1), b_mat.size(2)).to(device)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv
