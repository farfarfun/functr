# !/usr/bin/python
import torch
from torch import nn
import torch.nn.functional as F


class CoreCapsuleNetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        seq_len: int,
        interest_num: int = 4,
        routing_times: int = 3,
        hard_readout: bool = True,
        relu_layer: bool = False,
        *args,
        **kwargs,
    ):
        super(CoreCapsuleNetwork, self).__init__()
        self.hidden_size = hidden_size  # h
        self.seq_len = seq_len  # s
        self.interest_num = interest_num  # i
        self.routing_times = routing_times
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True
        self.relu = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False), nn.ReLU()
        )

    def forward(self, mask, item_eb_hat, capsule_weight, *args, **kwargs):
        item_eb_hat = torch.reshape(
            item_eb_hat, (-1, self.seq_len, self.interest_num, self.hidden_size)
        )
        item_eb_hat = torch.transpose(item_eb_hat, 2, 1)

        # shape=[b, i, s, h]
        if self.stop_grad:  # 截断反向传播，item_emb_hat不计入梯度计算中
            item_eb_hat_iter = item_eb_hat.detach()
        else:
            item_eb_hat_iter = item_eb_hat

        # 动态路由传播3次
        for i in range(self.routing_times):
            # [b, i, s]
            attention_mask = torch.repeat_interleave(
                torch.unsqueeze(mask, 1), self.interest_num, 1
            )
            paddings = torch.zeros_like(attention_mask)

            # 计算c，进行mask，最后shape=[b, i, 1, s]
            capsule_softmax_weight = F.softmax(capsule_weight)
            # capsule_softmax_weight = F.softmax(capsule_weight, axis=-1)
            capsule_softmax_weight = torch.where(
                attention_mask == 0, paddings, capsule_softmax_weight
            )

            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)

            if i < self.routing_times - 1:
                # s=c*u_hat , (b, i, 1, s) * (b, i, s, h)
                # shape=(b, i, 1, h)
                interest_capsule = torch.matmul(
                    capsule_softmax_weight, item_eb_hat_iter
                )

                # shape=(b, i, 1, 1)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, keepdim=True)

                # shape=(b, i, 1, 1)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)

                # squash(s)->v,shape=(b, i, 1, h)
                interest_capsule = scalar_factor * interest_capsule

                # 更新b

                # u_hat*v, shape=(b, i, s, 1)
                delta_weight = torch.matmul(
                    item_eb_hat_iter,
                    torch.transpose(interest_capsule, 3, 2),
                    # shape=(b, i, h, 1)
                )

                # shape=(b, i, s)
                delta_weight = torch.reshape(
                    delta_weight, (-1, self.interest_num, self.seq_len)
                )
                capsule_weight = capsule_weight + delta_weight  # 更新b
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, keepdim=True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = torch.reshape(
            interest_capsule, (-1, self.interest_num, self.hidden_size)
        )

        # MIND模型使用book数据库时，使用relu_layer
        if self.relu_layer:
            interest_capsule = self.relu(interest_capsule)
        return interest_capsule


class MINDCapsuleNetwork(CoreCapsuleNetwork):
    def __init__(self, *args, **kwargs):
        super(MINDCapsuleNetwork, self).__init__(*args, **kwargs)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, item_eb, mask, *args, **kwargs):
        # [b, s, h]
        item_eb_hat = self.linear(item_eb)
        # [b, s, h*in]
        item_eb_hat = torch.repeat_interleave(item_eb_hat, self.interest_num, 2)
        capsule_weight = torch.randn(
            (item_eb_hat.shape[0], self.interest_num, self.seq_len)
        )
        return super().forward(mask, item_eb_hat, capsule_weight)


class BCapsuleNetwork(CoreCapsuleNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(
            self.hidden_size, self.hidden_size * self.interest_num, bias=False
        )

    def forward(self, item_eb, mask, *args, **kwargs):
        item_eb_hat = self.linear(item_eb)
        capsule_weight = torch.zeros(
            (item_eb_hat.shape[0], self.interest_num, self.seq_len)
        )
        return super().forward(mask, item_eb_hat, capsule_weight)


class MIND(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_length: int,
        n_items: int,
        interest_num: int = 5,
        *args,
        **kwargs,
    ):
        super(MIND, self).__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.n_items = n_items

        self.item_emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        # capsule network
        self.capsule = MINDCapsuleNetwork(
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
