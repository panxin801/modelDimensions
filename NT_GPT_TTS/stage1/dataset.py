import sys
import random
import kaldiio
import torch.utils.data as data
import torch
import torch.nn.functional as F
import tqdm

sys.path.append("../")

from phone import symbols


class TransducerDataset(data.Dataset):
    def __init__(self, melScp, tokenScp, metaFile, mode, segmentsize):
        super().__init__()

        self.melDict = kaldiio.load_scp(melScp)
        self.tokenDict = kaldiio.load_scp(tokenScp)
        # self.bos_token = torch.LongTensor([0])  # <BOS>
        # self.eos_token = torch.LongTensor([513])
        # self.pad_token = torch.LongTensor([514]) # KMeans clustering has 512, these are included in the phone symbols.

        self.segmentSize = segmentsize
        self.mode = mode
        self.metaList = self.parse_meta_file(metaFile)

    def parse_meta_file(self, metaFile):
        self.phoneSet = symbols
        metaList = []

        phone_to_idx = {v: i for i, v in enumerate(self.phoneSet)}
        dirName = ['p261', 'p225', 'p294', 'p347', 'p238',
                   'p234', 'p248', 'p335', 'p245', 'p326', 'p302']

        with open(metaFile, "rt", encoding="utf8") as fr:
            metaLines = fr.read().splitlines()

        for line in tqdm.tqdm(metaLines):
            tempList = []

            uttid, feat = line.split("||")
            if ((uttid[0:4] in dirName and self.mode == 'valid') or (uttid[0:4] not in dirName and self.mode == 'train')):
                featList = feat.strip().split(" ")
                # print(uttid)
                # print(featList)

                tempList.append(uttid)
                tempList.append([phone_to_idx[phone]
                                for phone in featList])  # phoneme to int index

                metaList.append(tempList)

        return metaList

    def __len__(self):
        return len(self.metaList)

    def __getitem__(self, index):
        uttid, phones = self.metaList[index]  # phoines are int list

        tokenSeq = torch.from_numpy(self.tokenDict[uttid])  # [L_token]
        phoneIds = torch.LongTensor(phones)  # [L_phone]

        mels = torch.from_numpy(self.melDict[uttid])  # [L_mels, 80]
        melsStart = random.randint(0, mels.size(
            0) - self.segmentSize) if mels.size(0) > self.segmentSize else 0
        segmentMel = mels[melsStart:melsStart + self.segmentSize, :]
        segmentMel = segmentMel.permute(1, 0)  # [80, segmentSize]

        if segmentMel.size(1) < self.segmentSize:
            segmentMel = F.pad(
                segmentMel, (0, self.segmentSize - segmentMel.size(1)), "constant")

            assert segmentMel.size(1) == self.segmentSize
            assert segmentMel.size(0) == 80

        # segmentMel  may truncate, phone and token are not.
        # [L_phone], [L_token], [80, segmentSize]
        return (phoneIds, tokenSeq, segmentMel)


class TransducerCollate():
    def __init__(self, segmentsize):
        self.segmentsize = segmentsize

    def __call__(self, batch):
        """ phoneIds, tokenSeq, ssegmentMel = batch
            phoneIds: [L_phone] 
            tokenSeq: [L_token]
            ssegmentMel: [80, segmentSize]
        Return:
            phonePadded: [B,max_phone_len]
            tokenPadded: [B,max_token_len]
            melPadded: [B,80,segmentSize]
            phoneSeqLens: [B]
            tokenSeqLens: [B]
        """

        inputLens, idxSortedDescreasing = torch.sort(torch.LongTensor(
            [x[0].size(0) for x in batch]),
            dim=0,
            descending=True)  # inputLens是长度倒叙结果，idxSortedDescreasing是其在batch中的索引

        maxInputLen = inputLens[0]  # phone的最大长度
        phonePadded = torch.zeros(
            len(batch), maxInputLen, dtype=torch.long)  # [B,L_max_phone]

        maxTokenLen = max(x[1].size(0) for x in batch)  # token的最大长度
        tokenPadded = torch.zeros(
            len(batch), maxTokenLen, dtype=torch.long)  # [B,L_max_token]

        melPadded = torch.FloatTensor(
            len(batch), 80, self.segmentsize)  # [B,80,segmentSize]

        tokenSeqLens = torch.LongTensor(len(batch))
        phoneSeqLens = torch.LongTensor(len(batch))

        for i in range(len(idxSortedDescreasing)):
            phone = batch[idxSortedDescreasing[i]][0]
            phonePadded[i, :phone.size(0)] = phone
            phoneSeqLens[i] = phone.size(0)

            tokenSeq = batch[idxSortedDescreasing[i]][1]
            tokenPadded[i, :tokenSeq.size(0)] = tokenSeq
            tokenSeqLens[i] = tokenSeq.size(0)

            mel = batch[idxSortedDescreasing[i]][2]
            melPadded[i, :, :] = mel

        return (phonePadded, tokenPadded, melPadded, phoneSeqLens, tokenSeqLens)


if __name__ == "__main__":
    trainset = TransducerDataset(
        "mel16k.scp",
        "token.scp",
        "total.txt",
        "train",
        187)

    phoneIds, tokenSeq, segmentMel = trainset[0]
    print(phoneIds.size())
    print(tokenSeq.size())
    print(segmentMel.size())
    print(segmentMel)

    collateFn = TransducerCollate(187)

    dataloader = data.DataLoader(trainset,
                                 batch_size=18,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=4,
                                 collate_fn=collateFn,
                                 drop_last=False)
    # iter once
    for i, data in enumerate(dataloader):
        phonePadded, tokenPadded, melPadded, phoneSeqLens, tokenSeqLens = data
        print(phonePadded.size())
        print(tokenPadded.size())
        print(melPadded.size())
        print(phoneSeqLens)
        print(tokenSeqLens)
        break
