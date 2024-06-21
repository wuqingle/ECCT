"""
Deep Coding for Linear Block Error Correction
"""
import torch
import torch.nn as nn
import logging
def sign_to_bin(x):
    return 0.5 * (1 - x)

def bin_to_sign(x):
    return 1 - 2 * x

def diff_syndrome(H,x):
    tmp = bin_to_sign(H.unsqueeze(0)*x.unsqueeze(1))
    tmp = torch.prod(tmp,2)
    return sign_to_bin(tmp)

def diff_gener(G,m):
    tmp = bin_to_sign(G.unsqueeze(0)*m.unsqueeze(2))
    tmp = torch.prod(tmp,1)
    return sign_to_bin(tmp)

class Binarization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return ((input>=0)*1. - (input<0)*1.).float()
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return grad_output*(torch.abs(x)<=1)

class E2E_DC_ECC_Transformer(nn.Module):
    def __init__(self, args, decoder):
        super(E2E_DC_ECC_Transformer, self).__init__()
        ####
        self.args = args
        code = args.code
        self.n = code.n
        self.k = code.k
        self.bin = Binarization.apply
        with torch.no_grad():
            P_matrix = (torch.randint(0,2,(code.n-code.k,code.k))).float()
            P_matrix = bin_to_sign(P_matrix)*0.01
        self.P_matrix = nn.Parameter(P_matrix)
        # self.register_buffer('P_matrix', P_matrix)    
        self.register_buffer('I_matrix_H', torch.eye(code.n-code.k))
        self.register_buffer('I_matrix_G', torch.eye(code.k))
        #
        self.decoder = decoder
        ########
        
    def forward(self, m, z):
        x = diff_gener(self.get_generator_matrix(), m)
        x = bin_to_sign(x)
        z_mul = ((x+z) * x).detach()
        y = x*z_mul
        syndrome = bin_to_sign(diff_syndrome(self.get_pc_matrix(),sign_to_bin(self.bin(y))))
        magnitude = torch.abs(y)
        emb, loss, x_pred = self.decoder(magnitude, syndrome, self.get_pc_matrix(), z_mul, y, self.get_pc_matrix())
        return loss, x_pred, sign_to_bin(x)        
    
    def get_pc_matrix(self):
        bin_P =  sign_to_bin(self.bin(self.P_matrix))
        return torch.cat([self.I_matrix_H,bin_P],1)
    
    def get_generator_matrix(self,):
        bin_P =  sign_to_bin(self.bin(self.P_matrix))
        return torch.cat([bin_P,self.I_matrix_G],0).transpose(0,1)
import time
def test(model, device, EbNo_range_test, min_FER=100):
    model.eval()
    test_loss_list, test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], [], []
    t = time.time()
    with torch.no_grad():
        # for ii, test_loader in enumerate(test_loader_list):
        std_test = torch.tensor([EbN0_to_std(ebn0, code.k / code.n) for ebn0 in EbNo_range_test]).float()
        i=0
        # for ii in arange(EbNo_range_test):
        for ii in EbNo_range_test:
            # print(ii,EbNo_range_test,std_test)
            stds = std_test[i].expand(bs)#torch.randperm(bs) % len(std_test)
            i+=1
            test_loader_list = torch.randn(bs, code.n) * stds.unsqueeze(-1)
            test_loss = test_ber = test_fer = cum_count = 0.
            while True:
                # (m, x, z, y, magnitude, syndrome) = next(iter(test_loader))
                # z_mul = (y * bin_to_sign(x))
                # z_pred = model(magnitude.to(device), syndrome.to(device))
                # loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))
                loss, x_pred, x = model(m, test_loader_list[ii].to(device))

                test_loss += loss.item() * x.shape[0]

                test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                test_fer += FER(x_pred, x.to(device)) * x.shape[0]
                print("test_ber,test_fer,", test_ber,test_fer)
                cum_count += x.shape[0]#                              1e5 100         1e9 1000
                if (min_FER > 0 and test_fer > min_FER and cum_count >1e5) or cum_count >= 1e9:
                    if cum_count >= 1e9:
                        print(f'Number of samples threshold reached for EbN0:{ii}')#{EbNo_range_test[ii]}
                    else:
                        print(f'FER count threshold reached for EbN0:{ii}')#{EbNo_range_test[ii]
                    break
            # break
            cum_samples_all.append(cum_count)
            test_loss_list.append(test_loss / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            print(f'Test EbN0={ii}, BER={test_loss_ber_list[-1]:.2e}')#EbNo_range_test[ii]
        ###
        logging.info('\nTest Loss ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_list, EbNo_range_test))]))
        logging.info('Test FER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_fer_list, EbNo_range_test))]))
        logging.info('Test BER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ber_list, EbNo_range_test))]))
        # logging.info('Test -ln(BER) ' + ' '.join(
        #     ['{}: {:.2e}'.format(ebno, -np.log(elem)) for (elem, ebno)
        #      in
        #      (zip(test_loss_ber_list, EbNo_range_test))]))
    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_list, test_loss_ber_list, test_loss_fer_list
def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()
############################################################
############################################################

if __name__ == '__main__':
    from DC_ECCT import DC_ECC_Transformer
    import numpy as np
    class Code():
        pass
    def EbN0_to_std(EbN0, rate):
        snr =  EbN0 + 10. * np.log10(2 * rate)
        return np.sqrt(1. / (10. ** (snr / 10.)))
    code = Code()
    code.k = 16
    code.n = 31
    
    args = Code()
    args.code = code
    args.d_model = 32
    args.h = 8
    args.N_dec = 2
    args.dropout_attn = 0
    args.dropout = 0
    
    bs = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = E2E_DC_ECC_Transformer(args, DC_ECC_Transformer(args)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    EbNo_range_train = range(2, 8)
    EbNo_range_test = range(4, 6)#4,7
    # EbNo_range_train = range(2, 8)
    std_train = torch.tensor([EbN0_to_std(ebn0, code.k / code.n) for ebn0 in EbNo_range_train]).float()
    # test_dataloader_list=
    std_test = torch.tensor([EbN0_to_std(ebn0, code.k / code.n) for ebn0 in EbNo_range_test]).float()

    m = torch.ones((bs, code.k)).long().to(device)
    H0 = model.get_pc_matrix().detach().clone()
    for iter in range(10000):
        model.zero_grad()
        stds = std_train[torch.randperm(bs)%len(std_train)]
        loss, x_pred, x = model(m, (torch.randn(bs,code.n)*stds.unsqueeze(-1)).to(device))
        loss.backward()
        optimizer.step()
        if iter%1000 == 0:
            print(f'iter {iter}: loss = {loss.item()} BER = {torch.mean((x_pred!=x).float()).item()} ||H_t-H0||_1 = {torch.sum((H0-model.get_pc_matrix()).abs())}')
        if iter   == 9999:# 9999or epoch in [1, args.epochs]: 9999
            a,ber,fer=test(model, device, EbNo_range_test)
            # stds_t = std_test[torch.randperm(bs) % len(std_test)]
            # loss, x_pred, x = model(m, (torch.randn(bs, code.n) * stds_t.unsqueeze(-1)).to(device))
            print("ber,fer",ber,fer)

        
