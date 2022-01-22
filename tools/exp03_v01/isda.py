import torch
import torch.nn as nn
import torch.nn.functional as F


class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num):
        super(ISDALoss, self).__init__()

        self.class_num = class_num

        self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]

        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]

        sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(
            CV_temp.view(N, 1, A).expand(N, C, A)
        ).sum(2)

        aug_result = y + 0.5 * sigma2

        return aug_result

    def forward(self, fc, features, logits, labels, ratio, cv_matrix, manner):

        isda_aug_logits = self.isda_aug(fc, features, logits, labels, cv_matrix, ratio)

        loss = F.cross_entropy(isda_aug_logits, labels)
        
        return loss