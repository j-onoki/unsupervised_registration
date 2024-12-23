import model as m
import load_mnist as l
import torch
import matplotlib.pyplot as plt
import loss
import einops
from einops import rearrange

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = m.model().to(device)
    
    model.load_state_dict(torch.load("model300.pth"))


    #MNISTデータのダウンロード
    train_images = torch.load("./data/train_3_missing.pt")
    test_images = torch.load("./data/test_3_missing.pt")
    train_invert= torch.load("./data/train_3_invert.pt")
    test_invert= torch.load("./data/test_3_invert.pt")

    indexes = torch.randperm(test_invert.size()[0])
    test_invert = test_invert[indexes]

    #データセットの作成
    train_dataset = torch.utils.data.TensorDataset(train_images, train_invert)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_invert)
    m = test_dataset[4][0].reshape(1,1,32,32).to(device)
    f = test_dataset[4][1].reshape(1,1,32,32).to(device)
    
    with torch.no_grad(): 
        mphi, phi = model(m, f)
    
    phi2 = torch.zeros(1, 1, 32, 32).to(device)
    phi = torch.cat((phi2, phi), dim=1)
    phi = rearrange(phi, "b c h w -> h w (b c)")

    #画素値を正規化
    phi[:, :, 1] = (phi[:, :, 1] - torch.min(phi[:, :, 1]))/(torch.max(phi[:, :, 1])-torch.min(phi[:, :, 1]))
    phi[:, :, 2] = (phi[:, :, 2] - torch.min(phi[:, :, 2]))/(torch.max(phi[:, :, 2])-torch.min(phi[:, :, 2]))

    plt.figure()
    plt.imshow(f.cpu().reshape(32, 32), cmap = "gray")
    plt.savefig('./image/test_f.png')

    plt.figure()
    plt.imshow(m.cpu().reshape(32, 32), cmap = "gray")
    plt.savefig('./image/test_m.png')

    plt.figure()
    plt.imshow(mphi.cpu().reshape(32, 32), cmap = "gray")
    plt.savefig('./image/test_mphi.png')

    plt.figure()
    plt.imshow(phi.cpu())
    plt.savefig('./image/test_phi.png')