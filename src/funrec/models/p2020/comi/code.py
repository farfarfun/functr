from funrec.models.p2019.mind.core import CoreCapsuleNetwork
import torch
from torch import nn


class ComiCapsuleNetwork(CoreCapsuleNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = self.create_parameter(
            shape=[
                1,
                self.seq_len,
                self.interest_num * self.hidden_size,
                self.hidden_size,
            ]
        )

    def forward(self, item_eb, mask, *args, **kwargs):
        # shape=(batch_size, maxlen, 1, embedding_dim)
        u = torch.unsqueeze(item_eb, 2)
        # shape=(batch_size, maxlen, hidden_size*interest_num)
        item_eb_hat = torch.sum(self.w[:, : self.seq_len, :, :] * u, 3)
        capsule_weight = torch.zeros(
            (item_eb_hat.shape[0], self.interest_num, self.seq_len)
        )
        return super().forward(mask, item_eb_hat, capsule_weight)


class COMI(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_length: int,
        n_items: int,
        interest_num: int = 5,
        *args,
        **kwargs,
    ):
        super(COMI, self).__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.n_items = n_items

        self.item_emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        # capsule network
        self.capsule = ComiCapsuleNetwork(
            self.embedding_dim,
            self.max_length,
            interest_num=interest_num,
        )
        self.loss_fun = nn.CrossEntropyLoss()
        self.reset_parameters()

    def calculate_loss(self, user_emb, pos_item):
        all_items = self.item_emb.weight
        scores = torch.matmul(user_emb, all_items.transpose(1, 0))
        pos_item = pos_item.squeeze(1).long()
        return self.loss_fun(scores, pos_item)

    def output_items(self):
        return self.item_emb.weight

    def reset_parameters(self, initializer=None):
        for weight in self.parameters():
            if len(weight.shape) < 2:
                torch.nn.init.kaiming_normal_(weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_normal_(weight)

    def forward(self, item_seq, mask, item, train=True):
        if train:
            # 1. embedding layer
            item_seq = item_seq.long()
            seq_emb = self.item_emb(item_seq)  # Batch,Seq,Emb
            item_e = self.item_emb(item.long()).squeeze(1)

            # 2. multi-interest extractor layer + 3. label-aware attention layer
            multi_interest_emb = self.capsule(seq_emb, mask)  # Batch,K,Emb
            cos_res = torch.bmm(multi_interest_emb, item_e.squeeze(1).unsqueeze(-1))
            # 取内积结果最大的，作为最后的得分，并且取出对应的index
            k_index = torch.argmax(cos_res, dim=1)
            best_interest_emb = torch.rand(
                (multi_interest_emb.shape[0], multi_interest_emb.shape[2])
            )
            for k in range(multi_interest_emb.shape[0]):
                best_interest_emb[k, :] = multi_interest_emb[k, k_index[k], :]

            # 4. loss function
            loss = self.calculate_loss(best_interest_emb, item)
            output_dict = {
                "user_emb": multi_interest_emb,
                "loss": loss,
            }
        else:
            # test stage
            item_seq = item_seq.long()
            seq_emb = self.item_emb(item_seq)  # Batch,Seq,Emb
            multi_interest_emb = self.capsule(seq_emb, mask)  # Batch,K,Emb
            output_dict = {
                "user_emb": multi_interest_emb,
            }
        return output_dict
