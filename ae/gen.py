import os
import torch
from model import AutoEncoder
from torchvision.utils import save_image

def gen_pic(ckpt, epoch):
    device = torch.device('cuda')
    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    sample = torch.randn(64, 20).to(device)
    sample = model.decode(sample).cpu()
    save_image(sample.view(64, 1, 28, 28),
               'results/sample_e{:02}.png'.format(epoch))

if __name__ == '__main__':
    ckpts = os.listdir('ckpts')
    ckpts = [os.path.join('ckpts', ck) for ck in ckpts if not ck.startswith('.')]
    print(ckpts)
    
    e = 0
    for c in ckpts:
        e += 5
        gen_pic(c, e)