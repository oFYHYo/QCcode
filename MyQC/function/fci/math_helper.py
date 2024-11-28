import numpy as np
from itertools import combinations
from scipy import special
import time

def comb(nmo, nelec_s):
    ''' 计算组合数:C_nmo^nalpha or C_nmo^nbeta '''

    return int(special.comb(nmo, nelec_s, exact=True))

def gen_comb(nmo,nocc):
    ''' 生成给定轨道数和占据数下电子所有可能的排列方式 '''
    assert nocc <= nmo
    
    occ_positions = combinations(range(nmo), nocc)

    result = np.zeros((0, nmo), dtype=int)
    
    for pos in occ_positions:
        
        tmp = np.zeros(nmo, dtype=int)  
        tmp[list(pos)] = 1  
        
        result = np.vstack([result, tmp]) 
    
    return result

class Configure:
    ''' 该类代表一个组态 '''

    def __init__(self,occ_idx_alph,occ_idx_beta):
        assert len(occ_idx_alph) == len(occ_idx_beta)

        self.nelec = (occ_idx_alph + occ_idx_beta).sum()
        self.nmo = len(occ_idx_alph)
        self.occ_idx_alph = occ_idx_alph
        self.occ_idx_beta = occ_idx_beta

def gen_all_config(nmo,nalph,nbeta):
    ''' 生成所有可能的组态 '''

    alph_config = gen_comb(nmo,nalph)
    beta_config = gen_comb(nmo,nbeta)
    
    all_config = []
    for config_a in alph_config:
        for config_b in beta_config:
            all_config.append(Configure(config_a,config_b))
    
    return all_config

def diff_config(config1,config2):
    ''' 比较两组态之间的差异 
    
        diff_idx_n_s: 组态n与另一组态的s自旋电子占据数不同的位置 
        comm_idx_s  : 两组态s自旋电子占据数相同的位置  
        diff_num    : 两组态相差的激发数目。0代表为同一组态，1代表相差一个激发，以此类推
        '''

    config1_nmo = config1.nmo
    config2_nmo = config2.nmo
    assert config1_nmo == config2_nmo

    config1_alph = config1.occ_idx_alph
    config1_beta = config1.occ_idx_beta
    config2_alph = config2.occ_idx_alph
    config2_beta = config2.occ_idx_beta

    diff_alph = config1_alph-config2_alph
    diff_beta = config1_beta-config2_beta
    
    diff_idx_1_alph = []
    diff_idx_2_alph = []
    diff_idx_1_beta = []
    diff_idx_2_beta = []
    comm_idx_alph = []
    comm_idx_beta = []

    for i in range(config1_nmo):
        if diff_alph[i] == 1:
            diff_idx_1_alph.append(i)
        elif diff_alph[i] == -1:
            diff_idx_2_alph.append(i)
        elif diff_alph[i] == 0 and config1_alph[i] == 1:
            comm_idx_alph.append(i)

        
        if diff_beta[i] == 1:
            diff_idx_1_beta.append(i)
        elif diff_beta[i] == -1:
            diff_idx_2_beta.append(i)
        elif diff_beta[i] == 0 and config1_beta[i] == 1:
            comm_idx_beta.append(i)
    
    # 计算两组态相差轨道个数
    diff_num  = np.abs(config1_alph-config2_alph).sum()
    diff_num += np.abs(config1_beta-config2_beta).sum()
    diff_num  = int(diff_num/2)

    return comm_idx_alph, comm_idx_beta, diff_idx_1_alph, diff_idx_2_alph, diff_idx_1_beta, diff_idx_2_beta, diff_num



           
def Davidson_fci(matrix,num_eig = 1,max_iter = 50,conv = 1e-6):
    ''' Davidson对角化 '''

    x = time.time()
    
    hdiag = np.diagonal(matrix)
    k = 5+num_eig

    A_size = matrix.shape[0]

    eig = 2*k

    # 生成初猜矩阵
    V_old = np.zeros([A_size,eig])
    hdiag = hdiag.reshape(-1,)
    Dsort = hdiag.argsort()
    
    for j in range(k):
        V_old[Dsort[j], j] = 1.0

    
    print('iter       sub_size       |r|_max')
    for i in range(max_iter):
        
        # 获得子空间下的A矩阵
        W_old = np.einsum("ki,il->kl",matrix,V_old)
        #W_old = vind(V_old.T).T
        sub_A = np.einsum("ik,il->kl",V_old,W_old)
        
        upper = np.triu_indices(n=sub_A.shape[0], k=1)
        lower = (upper[1], upper[0])
        sub_A[lower] = sub_A[upper]

        val,ket = np.linalg.eigh(sub_A)
        
        sub_val = val[:k]
        sub_ket = ket[:,:k]

        # 计算残差
        residual = np.einsum("ki,il->kl",W_old,sub_ket) - np.einsum("l,ki,il->kl",sub_val,V_old,sub_ket)
        
        r_norm = np.linalg.norm(residual,axis=0).tolist()
        
        max_norm = np.max(r_norm)
        print(f'{i+1:3d}{sub_A.shape[1]:12}{max_norm:18.6f}')
        if i>2 and max_norm < conv:
            y = time.time()
            print(f'Davidson对角化完成，耗时{y-x:.6f}s')
            break
        
        # 计算新的初猜矢量
        t = 1e-8
        D = np.repeat(hdiag.reshape(-1,1), k, axis=1) - sub_val
        D = np.where( abs(D) < t, np.sign(D)*t, D)
        new_guess = residual/D

        # 正交化
        V_new = np.hstack([V_old,new_guess])
        V_old,_ = np.linalg.qr(V_new)

    else:
        raise ValueError('Davidson不收敛')
    
    sub_ket = np.einsum("ki,il->kl",V_old,sub_ket)*np.sqrt(.5)
    
    return sub_val, sub_ket


if __name__ == '__main__':
    x=time.time()
    configs = gen_all_config(8,4,4)

    
    for config1 in configs:
        for config2 in configs:
            a = diff_config(config1,config2)
            #if a[-1] ==1:
                #print()
                #print(config1.occ_idx_alph,config1.occ_idx_beta)
                #print(config2.occ_idx_alph,config2.occ_idx_beta)
                
                #print(a)
    y = time.time()
    print(y-x)