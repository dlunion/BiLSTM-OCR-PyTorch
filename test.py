import random

import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.utils.data
import torchvision.transforms.functional as F

import dbface
import utils


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

        # 这里的256 等于forward时下面的c * h
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
model.eval()

convert = utils.StrLabelConverter("ABCDEFGHIJKLMNOPQRSTUVWXYZ", ignore_case=False)
checkpoint = torch.load("model.pth")
model.load_state_dict(checkpoint)
dataloader = torch.utils.data.DataLoader(Dataset(), 1, True, num_workers=0)

for image, text in dataloader:
    rawImage = ((image[0].data.numpy() + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0)

    image = image.cuda()
    predict = model(image)

    _, predict = predict.max(2)
    predict = predict.squeeze()
    predict = predict.contiguous()
    result = convert.decode(predict, torch.IntTensor([predict.shape[0]]), True)
    print(result)

    cv2.imshow("rawImage", rawImage)
    cv2.waitKey()
