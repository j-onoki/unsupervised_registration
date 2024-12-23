import torch
from tqdm import tqdm
import loss

#1epoch分の訓練を行う関数
def train_model(model, train_loader, optimizer, device):

    train_loss = 0.0

    # 学習モデルに変換
    model.train()


    for i, (f, m) in enumerate(tqdm(train_loader)):

        f, m = f.to(device), m.to(device)
        f = f.reshape(f.size()[0], 1, 32, 32)
        m = m.reshape(m.size()[0], 1, 32, 32)

        # 勾配を初期化
        optimizer.zero_grad()

        # 推論(順伝播)
        mphi, phi = model(m, f)

        # 損失の算出
        loss = criterion(mphi, f, phi)
        
        # 誤差逆伝播
        loss.backward()

        # パラメータの更新
        optimizer.step()

        # lossを加算
        train_loss += loss.item()
    
    # lossの平均値を取る
    train_loss = train_loss / len(train_loader)
    
    return train_loss

#モデル評価を行う関数
def test_model(model, test_loader, optimizer, device):

    test_loss = 0.0

    # modelを評価モードに変更
    model.eval()

    with torch.no_grad(): # 勾配計算の無効化

        for i, (f, m) in enumerate(tqdm(test_loader)):

            f, m = f.to(device), m.to(device)
            f = f.reshape(f.size()[0], 1, 32, 32)
            m = m.reshape(m.size()[0], 1, 32, 32)

            mphi, phi = model(m, f)
            loss = criterion(mphi, f, phi)
            
            test_loss += loss.item()


    # lossの平均値を取る
    test_loss = test_loss / len(test_loader)

    return test_loss

def lerning(model, train_loader, test_loader, optimizer, num_epochs, device):

    train_loss_list = []
    test_loss_list = []

    # epoch数分繰り返す
    for epoch in range(1, num_epochs+1, 1):

        train_loss = train_model(model, train_loader, optimizer, device)
        test_loss = test_model(model, test_loader, optimizer, device)
        
        print("epoch : {}, train_loss : {:.5f}, test_loss : {:.5f}" .format(epoch, train_loss, test_loss))

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        if epoch%100 == 0:
            filename = 'model' + str(epoch) + '.pth'
            torch.save(model.state_dict(), filename)
    return train_loss_list, test_loss_list

#cross correlation
def crossCorrelation(mphi, f):
    mphimean = torch.mean(mphi, dim=2, keepdim=True)
    mphimean = torch.mean(mphimean, dim=3, keepdim=True)
    fmean = torch.mean(f, dim=2, keepdim=True)
    fmean = torch.mean(fmean, dim=3, keepdim=True)
    mphi = mphi - mphimean
    f = f - fmean
    a = torch.sum(mphi*f, dim=2)
    a = torch.sum(a, dim=2)
    b1 = torch.sum(mphi*mphi, dim=2)
    b1 = torch.sqrt(torch.sum(b1, dim=2))
    b2 = torch.sum(f*f, dim=2)
    b2 = torch.sqrt(torch.sum(b2, dim=2))
    b = b1*b2
    cc = a/(b+1e-5)
    loss = torch.mean(cc)
    return loss

#損失関数
def criterion(mphi, f, phi):
    ramdaphi = 1
    g = loss.gradientLoss()
    cc = crossCorrelation(mphi, f)
    l = -cc*cc + ramdaphi*g(phi)
    return l
