from .unet_parts import *
from .convgru import ConvGRU


class TemporalUNet(nn.Module):
    def __init__(self, rnn, n_channels, n_classes, bilinear=True, kernel_size=(3, 3), num_layers=2):
        super(TemporalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.rnn_inc = rnn(input_dim=64, hidden_dim=64, kernel_size=kernel_size, num_layers=num_layers,
                           batch_first=True, bias=True, return_all_layers=False)
        self.down1 = Down(64, 128)
        self.rnn1 = rnn(input_dim=128, hidden_dim=128, kernel_size=kernel_size, num_layers=num_layers,
                        batch_first=True, bias=True, return_all_layers=False)

        self.down2 = Down(128, 256)
        self.rnn2 = rnn(input_dim=256, hidden_dim=256, kernel_size=kernel_size, num_layers=num_layers,
                        batch_first=True, bias=True, return_all_layers=False)

        self.down3 = Down(256, 512)
        self.rnn3 = rnn(input_dim=512, hidden_dim=512, kernel_size=kernel_size, num_layers=num_layers,
                        batch_first=True, bias=True, return_all_layers=False)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.rnn4 = rnn(input_dim=1024 // factor, hidden_dim=1024 // factor, kernel_size=kernel_size,
                        num_layers=num_layers, batch_first=True, bias=True, return_all_layers=False)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):  # X: BxTxCxHxW
        # print("model input shape: {}".format(x.shape))
        B, T, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x1 = self.inc(x)  # X1: BTxCxHxW

        BT, C, H, W = x1.shape
        x1t = x1.view(BT//T, T, C, H, W)
        # print("rnn input shape: {}".format(x.shape))
        f1, _ = self.rnn_inc(x1t)
        f1 = f1[:, -1, ::]
        # print("f1 shape: {}".format(f1.shape))

        x2 = self.down1(x1)
        BT, C, H, W = x2.shape
        x2t = x2.view(BT//T, T, C, H, W)
        f2, _ = self.rnn1(x2t)
        f2 = f2[:, -1, ::]
        # print("f2 shape: {}".format(f2.shape))

        x3 = self.down2(x2)
        BT, C, H, W = x3.shape
        x3t = x3.view(BT//T, T, C, H, W)
        f3, _ = self.rnn2(x3t)
        f3 = f3[:, -1, ::]
        # print("f3 shape: {}".format(f3.shape))

        x4 = self.down3(x3)
        BT, C, H, W = x4.shape
        x4t = x4.view(BT//T, T, C, H, W)
        f4, _ = self.rnn3(x4t)
        f4 = f4[:, -1, ::]
        # print("f4 shape: {}".format(f4.shape))

        x5 = self.down4(x4)
        BT, C, H, W = x5.shape
        x5t = x5.view(BT//T, T, C, H, W)
        f5, _ = self.rnn4(x5t)
        f5 = f5[:, -1, ::]
        # print("f5 shape: {}".format(f5.shape))

        x = self.up1(f5, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    channels = 1
    classes = 4
    hidden_dim = [64, 64, 128]

    model = TemporalUNet(ConvGRU, channels, classes, bilinear=True, kernel_size=(3, 3), num_layers=2)

    model = model.cuda()
    print(repr(model))
