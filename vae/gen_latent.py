import torch
import numpy as np
from torchvision.utils import save_image
from model import VAE

device = torch.device('cpu')
print('Using', device)

# == Model ==
model = VAE().to(device)
load_ckpt = True
if load_ckpt:
    ckpt_file = 'ckpts/e070vae_mnist.pt'
    ckpt  = torch.load(ckpt_file)
    model.load_state_dict(ckpt)

# get image
x = torch.randn(1, 1, 28, 28)
out, mu, logvar = model(x)
print(out.shape)
print(mu)

save_image(out.view(1, 28, 28), 'out.png')


# make to size 20 * 20 images form q(z) of distributed
imgs = np.empty((28 * 20, 28 * 20))
index_x = 0
for x in range(-10,10,1):
    index_y = 0
    for y in range(-10,10,1):
        value = np.array([[float(x / 5.), float(y / 5.)]])
        img = session.run(decoder_op, feed_dict={data_x:value})
        imgs[index_x * 28:(index_x + 1) * 28, index_y * 28:(index_y + 1) * 28] = img.reshape((28, 28))
        index_y += 1
    index_x += 1

# plt.imshow(imgs, cmap=plt.get_cmap('gray'))
# plt.show()

save_image(imgs, 'samples.png')