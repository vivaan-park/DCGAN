import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

root = "data"
workers = 2  # 워커 쓰레드 수
batch_size = 128
image_size = 64
nc = 3  # 이미지 채널 수(RGB)
nz = 100  # 잠재공간 벡터 크기
ngf = 64  # 생성자를 통과할 때 만들어지는 특징 데이터 채널 수
ndf = 64  # 구분자를 통과할 때 만들어지는 특징 데이터 채널 수
num_epochs = 5
lr = 0.0002
beta1 = 0.5


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 입력데이터가 가장 처음 통과하는 전치 합성곱 계층
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기 '(ngf*8) x 4 x 4'
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf* 4),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기 (ngf*4) x 8 x 8'
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기 '(ngf*2) x 16 x 16'
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기 '(ngf) x 32 x 32'
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 위의 계층을 통과한 데이터의 크기 '(nc) x 64 x 64'
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 입력 데이터 크기 '(nc) x 64 x 64'
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기 '(ndf) x 32 x 32'
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기 '(ndf*2) x 16 x 16'
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기 '(ndf*4) x 8 x 8'
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기 (ndf*8) x 4 x 4'
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

def main():
    device = {
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    }

    torch.manual_seed(42)

    dataset = dset.ImageFolder(root=root,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    net_gen = Generator().to(device)
    net_gen.apply(weights_init)

    net_dis = Discriminator().to(device)
    net_dis.apply(weights_init)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    real_label = 1.
    fake_label = 0.

    optimizer_gen = optim.Adam(net_gen.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_dis = optim.Adam(net_dis.parameters(), lr=lr, betas=(beta1, 0.999))

    # 학습상태 체크
    img_list = []
    gen_losses = []
    dis_losses = []
    iters = 0

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Discriminator 신경망 업데이트
            net_dis.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = net_dis(real_cpu).view(-1)
            err_dis_real = criterion(output, label)
            err_dis_real.backward()
            dis_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = net_gen(noise)
            label.fill_(fake_label)
            output = net_gen(fake.detach()).view(-1)
            err_dis_fake = criterion(output, label)
            err_dis_fake.backward()
            dis_gen_z1 = output.mean().item()
            err_dis = err_dis_real + err_dis_fake
            optimizer_dis.step()

            # Generator 신경망 업데이트
            net_gen.zero_grad()
            label.fill_(real_label)
            output = net_dis(fake).view(-1)
            err_gen = criterion(output, label)
            err_gen.backward()
            dis_gen_z2 = output.mean().item()
            optimizer_gen.step()

            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\t'
                      f'Discriminator loss: {err_dis.item()}\t'
                      f'Generator loss: {err_gen.item()}\t'
                      f'Discriminator(Generator(z)): {dis_gen_z1}/{dis_gen_z2}')

            # fixed_noise를 통과시킨 G의 출력값을 저장해둡니다
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = net_gen(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    main()
