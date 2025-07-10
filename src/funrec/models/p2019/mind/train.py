import torch


from torch.utils.data import Dataset, DataLoader


import pandas as pd
import numpy as np
import os
import math
import random
from sklearn.preprocessing import normalize
from tqdm import tqdm
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from .core import MIND


class SeqnenceDataset(Dataset):
    def __init__(self, config, df, phase="train"):
        self.config = config
        self.df = df
        self.max_length = self.config["max_length"]
        self.df = self.df.sort_values(by=["user_id", "timestamp"])
        self.user2item = self.df.groupby("user_id")["item_id"].apply(list).to_dict()
        self.user_list = self.df["user_id"].unique()
        self.phase = phase

    def __len__(
        self,
    ):
        return len(self.user2item)

    def __getitem__(self, index):
        if self.phase == "train":
            user_id = self.user_list[index]
            item_list = self.user2item[user_id]
            hist_item_list = []
            hist_mask_list = []

            k = random.choice(
                range(4, len(item_list))
            )  # 从[8,len(item_list))中随机选择一个index
            # k = np.random.randint(2,len(item_list))
            item_id = item_list[k]  # 该index对应的item加入item_id_list

            if k >= self.max_length:  # 选取seq_len个物品
                hist_item_list.append(item_list[k - self.max_length : k])
                hist_mask_list.append([1.0] * self.max_length)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.max_length - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.max_length - k))

            return (
                torch.Tensor(hist_item_list).squeeze(0),
                torch.Tensor(hist_mask_list).squeeze(0),
                torch.Tensor([item_id]),
            )
        else:
            user_id = self.user_list[index]
            item_list = self.user2item[user_id]
            hist_item_list = []
            hist_mask_list = []

            k = int(0.8 * len(item_list))
            # k = len(item_list)-1

            if k >= self.max_length:  # 选取seq_len个物品
                hist_item_list.append(item_list[k - self.max_length : k])
                hist_mask_list.append([1.0] * self.max_length)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.max_length - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.max_length - k))

            return (
                torch.Tensor(hist_item_list).squeeze(0),
                torch.Tensor(hist_mask_list).squeeze(0),
                item_list[k:],
            )

    def get_test_gd(self):
        self.test_gd = {}
        for user in self.user2item:
            item_list = self.user2item[user]
            test_item_index = int(0.8 * len(item_list))
            self.test_gd[user] = item_list[test_item_index:]
        return self.test_gd


config = {
    "train_path": "./data/data173799/train_enc.csv",
    "valid_path": "./data/data173799/valid_enc.csv",
    "test_path": "./data/data173799/test_enc.csv",
    "lr": 1e-4,
    "Epoch": 5,
    "batch_size": 256,
    "embedding_dim": 16,
    "num_layers": 1,
    "max_length": 20,
    "n_items": 15406,
    "K": 4,
}


def my_collate(batch):
    hist_item, hist_mask, item_list = list(zip(*batch))

    hist_item = [x.unsqueeze(0) for x in hist_item]
    hist_mask = [x.unsqueeze(0) for x in hist_mask]

    hist_item = torch.cat(hist_item, axis=0)
    hist_mask = torch.cat(hist_mask, axis=0)
    return hist_item, hist_mask, item_list


def save_model(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), path + "model.pth")


def load_model(model, path):
    state_dict = torch.load(path + "model.pth")
    model.load_state_dict(state_dict)
    print("model loaded from %s" % path)
    return model


"""
note: 基于faiss的向量召回
"""


def get_predict(model, test_data, hidden_size, topN=20):
    import faiss

    item_embs = model.output_items().cpu().detach().numpy()
    item_embs = normalize(item_embs, norm="l2")
    gpu_index = faiss.IndexFlatIP(hidden_size)
    gpu_index.add(item_embs)

    test_gd = dict()
    preds = dict()

    user_id = 0

    for item_seq, mask, targets in tqdm(test_data):
        # 获取用户嵌入
        # 多兴趣模型，shape=(batch_size, num_interest, embedding_dim)
        # 其他模型，shape=(batch_size, embedding_dim)
        user_embs = model(item_seq, mask, None, train=False)["user_emb"]
        user_embs = user_embs.cpu().detach().numpy()

        # 用内积来近邻搜索，实际是内积的值越大，向量越近（越相似）
        if len(user_embs.shape) == 2:  # 非多兴趣模型评估
            user_embs = normalize(user_embs, norm="l2").astype("float32")
            D, I = gpu_index.search(
                user_embs, topN
            )  # Inner Product近邻搜索，D为distance，I是index
            #             D,I = faiss.knn(user_embs, item_embs, topN,metric=faiss.METRIC_INNER_PRODUCT)
            for i, iid_list in enumerate(
                targets
            ):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                test_gd[user_id] = iid_list
                preds[user_id] = I[i, :]
                user_id += 1
        else:  # 多兴趣模型评估
            ni = user_embs.shape[1]  # num_interest
            user_embs = np.reshape(
                user_embs, [-1, user_embs.shape[-1]]
            )  # shape=(batch_size*num_interest, embedding_dim)
            user_embs = normalize(user_embs, norm="l2").astype("float32")
            # Inner Product近邻搜索，D为distance，I是index
            D, I = gpu_index.search(user_embs, topN)
            #             D,I = faiss.knn(user_embs, item_embs, topN,metric=faiss.METRIC_INNER_PRODUCT)

            # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
            for i, iid_list in enumerate(targets):
                recall = 0
                dcg = 0.0
                item_list_set = []

                # 将num_interest个兴趣向量的所有topN近邻物品（num_interest*topN个物品）集合起来按照距离重新排序
                item_list = list(
                    zip(
                        np.reshape(I[i * ni : (i + 1) * ni], -1),
                        np.reshape(D[i * ni : (i + 1) * ni], -1),
                    )
                )
                # 降序排序，内积越大，向量越近
                item_list.sort(key=lambda x: x[1], reverse=True)
                # 按距离由近到远遍历推荐物品列表，最后选出最近的topN个物品作为最终的推荐物品
                for j in range(len(item_list)):
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.append(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break
                test_gd[user_id] = iid_list
                preds[user_id] = item_list_set
                user_id += 1
    return test_gd, preds


def evaluate(preds, test_gd, topN=50):
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    for user in test_gd.keys():
        recall = 0
        dcg = 0.0
        item_list = test_gd[user]
        for no, item_id in enumerate(item_list):
            if item_id in preds[user][:topN]:
                recall += 1
                dcg += 1.0 / math.log(no + 2, 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no + 2, 2)
        total_recall += recall * 1.0 / len(item_list)
        if recall > 0:
            total_ndcg += dcg / idcg
            total_hitrate += 1
    total = len(test_gd)
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    return {f"recall@{topN}": recall, f"ndcg@{topN}": ndcg, f"hitrate@{topN}": hitrate}


# 指标计算
def evaluate_model(model, test_loader, embedding_dim, topN=20):
    test_gd, preds = get_predict(model, test_loader, embedding_dim, topN=topN)
    return evaluate(preds, test_gd, topN=topN)


# 读取数据
train_df = pd.read_csv(config["train_path"])
valid_df = pd.read_csv(config["valid_path"])
test_df = pd.read_csv(config["test_path"])
train_dataset = SeqnenceDataset(config, train_df, phase="train")
valid_dataset = SeqnenceDataset(config, valid_df, phase="test")
test_dataset = SeqnenceDataset(config, test_df, phase="test")
train_loader = DataLoader(
    dataset=train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    collate_fn=my_collate,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    collate_fn=my_collate,
)


model = MIND(config)
# model = SRGNN(config)
optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"])
# optimizer = torch.optimizer.Adam(parameters=model.parameters(), learning_rate=config['lr'])
log_df = pd.DataFrame()
best_reacall = -1

exp_path = "./ml-20m_softmax/MIND_{}_{}_{}/".format(
    config["lr"], config["batch_size"], config["embedding_dim"]
)
os.makedirs(exp_path, exist_ok=True, mode=0o777)
patience = 5
last_improve_epoch = 1
log_csv = exp_path + "log.csv"
# *****************************************************train*********************************************
for epoch in range(1, 1 + config["Epoch"]):
    # try :
    pbar = tqdm(train_loader)
    model.train()
    loss_list = []
    acc_50_list = []
    print()
    print("Training:")
    print()
    for batch_data in pbar:
        (item_seq, mask, item) = batch_data

        output_dict = model(item_seq, mask, item)
        loss = output_dict["loss"]

        loss.backward()
        optimizer.step()
        # optimizer.clear_grad()
        optimizer.zero_grad()

        loss_list.append(loss.item())

        pbar.set_description("Epoch [{}/{}]".format(epoch, config["Epoch"]))
        pbar.set_postfix(loss=np.mean(loss_list))
    # *****************************************************valid*********************************************

    print("Valid")
    recall_metric = evaluate_model(
        model, valid_loader, config["embedding_dim"], topN=50
    )
    print(recall_metric)
    recall_metric["phase"] = "valid"
    log_df = log_df.append(recall_metric, ignore_index=True)
    log_df.to_csv(log_csv)

    if recall_metric["recall@50"] > best_reacall:
        save_model(model, exp_path)
        best_reacall = recall_metric["recall@50"]
        last_improve_epoch = epoch

    if epoch - last_improve_epoch > patience:
        break

print("Testing")
model = load_model(model, exp_path)
recall_metric = evaluate_model(model, test_loader, config["embedding_dim"], topN=50)
print(recall_metric)
recall_metric["phase"] = "test"
log_df = log_df.append(recall_metric, ignore_index=True)
log_df.to_csv(log_csv)


# embedding分布可视化
def plot_embedding(data, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(dpi=120)
    plt.scatter(data[:, 0], data[:, 1], marker=".")

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


item_emb = model.output_items().detach().numpy()
tsne_emb = TSNE(n_components=2).fit_transform(item_emb)
plot_embedding(tsne_emb, "MIND Item Embedding")
