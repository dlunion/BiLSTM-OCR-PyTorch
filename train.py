import random

import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.utils.data
import torchvision.transforms.functional as F

import dbface
import utils
from warpctc import CTCLoss


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class Dataset:

    def __getitem__(self, item):
        return self.genImage()

    def __len__(self):
        return 10000

    def genImage(self):

        image = np.zeros((32, 128, 3), np.uint8)
        text = ""
        n = random.randint(1, 4)
        remain = 128 - n * 32
        x = -20 + random.randint(0, remain)
        for i in range(n):
            index = random.randint(1, 26)
            txt = chr(index - 1 + ord('A'))
            text += txt
            x += 20 + random.randint(0, 15)
            cv2.putText(image, txt, (x, 23), 0, 0.8, (255, 255, 255), 1, 16)

        image = (image / 255 - 0.5).astype(np.float32)
        return F.to_tensor(image), text


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.bone = dbface.Mbv3SmallFast()
        self.bone.load_pretrain()

        self.rnn1 = BidirectionalLSTM(256, 256, 27)
        #self.rnn2 = BidirectionalLSTM(64, 64, 27)

    def forward(self, input):
        x = self.bone(input)
        x = x.permute(3, 0, 1, 2)
        w, b, c, h = x.shape
        x = x.view(w, b, c * h)

        x = self.rnn1(x)
        #x = self.rnn2(x)
        return x

model = Model()
model.cuda()
lossFunc = CTCLoss()
convert = utils.StrLabelConverter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
#op = torch.optim.SGD(model.parameters(), 1e-3, 0.9)
op = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
dataloader = torch.utils.data.DataLoader(Dataset(), 64, True, num_workers=0)

for epoch in range(50):
    for index, (images, texts) in enumerate(dataloader):
        images = images.cuda()
        batch_size = int(images.shape[0])

        preds = model(images)
        preds_size = torch.IntTensor([int(preds.size(0))] * batch_size)
        t, l = convert.encode(texts)
        loss = lossFunc(preds, t, preds_size, l) / batch_size

        print(f"{epoch}. {index}, loss: {loss.item():.2f}")

        op.zero_grad()
        loss.backward()
        op.step()

torch.save(model.state_dict(), "model.pth")