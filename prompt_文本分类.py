# -*- coding: utf-8 -*-
"""
【方案一】基于「硬提示模板 + Masked Language Model」的少样本文本分类
====================================================================
整体思路（Prompt-based Learning / 提示学习）：

1. 传统分类是「喂入句子 -> 输出类别概率」，需要大量标注数据微调。
   本方案不同：借用预训练好的语言模型（中文 RoBERTa）本身「完形填空」的能力，
   人为构造一个提示模板让模型去「补全」。

2. 模板形如：
      文章内容为：<原始文本> 类别是：[MASK][MASK]
   其中 [MASK] 的个数 = 标签词被词表切分后的 token 个数（例如「财经」可能被切成 2 个 token）。

3. 训练时，模型输出的 logits 只在 [MASK] 位置参与损失计算（其它位置用 ignore_index=0 忽略），
   目标是让 [MASK] 位置预测出「标签词」对应的 token id。

4. 推理时，看 [MASK] 位置预测出的 token 是否等于真实标签词的 token，
   从而判断分类是否正确（见 cal_acc1 / cal_acc2）。

一句话总结：不直接学「文本->类别」映射，而是学「补全模板中的类别词」，
充分利用预训练知识，适合标注数据较少的场景。
"""

import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import random
from transformers import BertTokenizer
from transformers import BertModel, AutoModelForMaskedLM  # 后者是带 MLM 预测头的模型，能直接输出词表维度的 logits


def read_data(train_or_test, num=None):
    """
    读取数据集文件。

    参数:
        train_or_test : 字符串，'train' 或 'test'，用于拼接文件路径 data/{train_or_test}.txt
        num           : 可选，若指定则随机抽取 num 条样本（用于快速实验/少样本）

    返回:
        texts  : 文本列表
        labels : 标签列表（每个元素是字符串形式的类别索引，如 '3'）

    数据文件每行格式：  "原始文本\t类别索引"
    """
    # 打开 data 目录下对应的 txt 文件
    with open(os.path.join("data", train_or_test + ".txt"), encoding="utf-8") as f:
        all_data = f.read().split("\n")  # 按换行拆成一行一条

    texts = []
    labels = []
    for data in all_data:
        if data:  # 跳过空行
            t, l = data.split("\t")  # 文本和标签之间用 \t 分隔
            texts.append(t)
            labels.append(l)

    if num == None:
        return texts, labels
    else:
        # 随机无放回抽样 num 条
        sm_idx = random.sample(range(0, len(texts)), num)
        return [texts[i] for i in sm_idx], [labels[i] for i in sm_idx]


class TextDataset(Dataset):
    """数据集类：负责把「原始文本 + 标签」变成模型输入。

    注意：本类只负责单条样本的读取（__getitem__），
    真正「拼模板 + 转 token id」的重活在 collate_fn（pro_batch_data）里完成。
    """

    def __init__(self, texts, labels, tokenizer, index_2_label):
        self.texts = texts
        self.labels = labels
        # 模板里固定文字部分的长度，用于后面估算最大序列长度（预留 mask / 特殊 token 空间）
        # "文章内容为：" 共 6 字符，"类别是：" 共 4 字符，合计 10
        self.pro_len = len("文章内容为：" + "类别是：")

        self.tokenizer = tokenizer
        self.index_2_label = index_2_label  # 下标类别索引 -> 标签词文本（如 '财经'）

    def __getitem__(self, index):
        # 取第 index 条文本，只保留前 100 个字符，避免过长
        text = self.texts[index][:100]
        label = int(self.labels[index])  # 标签字符串 -> int 类别索引
        t_len = len(text)  # 当前文本实际长度（后面用于计算本 batch 的最大长度）
        # 这里只返回原始文本+标签，模板拼接放到 collate_fn 里按 batch 统一处理
        return text, label, t_len

    def __len__(self):
        return len(self.labels)

    def pro_batch_data(self, batch_data):
        """
        collate_fn：把同一个 batch 的若干样本拼成一份张量输入。

        关键逻辑 —— 构造提示模板：
            text_pro = "文章内容为：" + 文本 + "类别是：" + "[MASK] * k"
        其中 k = 标签词被切分后的 token 数量（len(index_2_label[label]) 之后要先 encode 才知道）。
        实际上这里 k 取的是「标签词文本的字符数」作为 [MASK] 占位个数。

        然后构建两个序列：
            text_idx  : 真正的模型输入（模板 + CLS/SEP）
            label_idx : 监督目标，只在 [MASK] 位置放真实标签词的 token id，其余位置全 0
                       （0 是损失的 ignore_index，会被忽略不参与计算）
        """
        # 解包：batch 里每条样本的 (text, label, len)
        batch_text, batch_label, batch_len = zip(*batch_data)
        # 计算本 batch 最大长度：最大文本长度 + 模板固定文字 + 额外预留的 CLS/SEP 等特殊 token
        max_len = max(batch_len) + self.pro_len + 2 + 2

        batch_text_idx = []
        batch_label_idx = []
        batch_text_len = []

        for text, label, len_ in zip(batch_text, batch_label, batch_len):
            # 1) 拼出提示模板：文本 + 类别占位 [MASK] * (标签词字符数)
            text_pro = "文章内容为：" + text + "类别是：" + "[MASK]" * len(self.index_2_label[label])

            # 2) encode 成 token id，add_special_tokens=True 会自动加上 [CLS] 和 [SEP]
            text_idx = self.tokenizer.encode(text_pro, add_special_tokens=True)
            # 记录去掉首尾 [CLS]/[SEP] 后的真实长度（评估时用）
            batch_text_len.append(len(text_idx) - 2)

            # 3) 构造监督标签序列 label_idx
            #    序列构成 = [0](CLS位置) + [0]*(中间全部忽略) + 标签词的 token id + [0](SEP位置)
            #    解释：本数据集每个中文标签词（如「财经」）在 RoBERTa 词表里会被切成 2 个 token，
            #          因此「len(text_idx) - 4」= 去掉 [CLS](1) + 2 个标签 token + [SEP](1) 共 4 个位置，
            #          中间的其它位置全部填 0（意味着损失计算时被忽略）。
            label_idx = [0] + [0] * (len(text_idx) - 4) + \
                self.tokenizer.encode(self.index_2_label[label], add_special_tokens=False) + [0]

            # 4) 把两条序列都 padding 到本 batch 的 max_len（不足的位置补 0）
            text_idx += [0] * (max_len - len(text_idx))
            label_idx += [0] * (max_len - len(label_idx))

            # 校验两条序列长度一致
            assert len(text_idx) == len(label_idx)

            batch_text_idx.append(text_idx)
            batch_label_idx.append(label_idx)

        # 转成张量返回（batch_text_len 保持 list，评估时用）
        return torch.tensor(batch_text_idx), torch.tensor(batch_label_idx), batch_text_len


class BertClassifier(nn.Module):
    """模型类：加载中文 RoBERTa 的 MLM 模型，冻结大部分参数，只训练输出头附近参数。

    为什么用 AutoModelForMaskedLM？
        —— 因为它自带「词表级」预测头，输出 shape 为 (batch, seq_len, vocab_size)，
           正是做「完形填空」需要的，我们直接在 [MASK] 位置取 logits 即可。
    """

    def __init__(self, model_name):
        super().__init__()
        # 加载带 MLM 头的预训练模型
        # self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.bert = AutoModelForMaskedLM.from_pretrained(model_name)

        # 冻结策略：只微调名字里含 "cls" 的参数（即 MLM 输出头），其余（BERT 主体）全部冻结
        # 这样可以极大减少训练参数量、防止过拟合。
        for name, param in self.bert.named_parameters():
            if "cls" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 交叉熵损失，ignore_index=0 表示 label 里值为 0 的位置不参与损失（即非 [MASK] 位置）
        self.loss_fun = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, batch_x, batch_len, label=None):
        """
        前向计算。

        参数:
            batch_x  : 输入 token id，shape (batch, seq_len)
            batch_len: 每句去掉 CLS/SEP 后的真实长度（本方案 forward 内暂未用到，主要为接口统一）
            label    : 训练时传入监督标签；为 None 表示推理

        返回:
            训练时返回 loss；推理时返回词表维度的 logits
        """
        # attention_mask = (batch_x > 0) 把 padding(0) 位置遮住，不让模型关注
        # 返回值取 [0] 即 logits，shape (batch, seq_len, vocab_size)
        pre = self.bert(batch_x, attention_mask=(batch_x > 0), return_dict=False)[0]

        if label is not None:
            # reshape 成 2 维，方便和 label(也 flatten) 一起送入交叉熵
            loss = self.loss_fun(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))
            return loss
        else:
            return pre  # 推理时返回完整 logits
            # return torch.argmax(pre, dim=-1)  # （另一种选择：直接返回每个位置的预测 token id）


# ---------------------------------------------------------------------------
# 下面是评估/调试用的辅助函数，注意它们会引用主程序里定义的全局 dict_ver
# ---------------------------------------------------------------------------

def cal__(all_pre, all_lab, all_len):
    """
    调试用统计函数：统计两种评估口径各自的对/错，遇到 acc1 对而 acc2 错时停下来打印，
    方便人工分析两种评估口径不一致的样本。
    """
    right1 = 0
    right2 = 0
    d_lens = sum([len(i) for i in all_pre])  # 总样本数

    # 从 dict_ver 的 key（形如 "6568-5307"）里拆出「两个标签 token id」，
    # 这是方案二的评估方式需要用到的：把两个位置的 logits 相加再取最大。
    k0, k1 = zip(*[i.split("-") for i in list(dict_ver.keys())])
    k0, k1 = torch.tensor([int(i) for i in k0]), torch.tensor([int(i) for i in k1])

    for pres, labs, lens in zip(all_pre, all_lab, all_len):
        for p, lab, len_ in zip(pres, labs, lens):
            # 口径1：逐位置 argmax，看两个 mask 位置预测的 token 是否与真实标签 token 完全一致
            p1 = torch.argmax(p, dim=-1)
            r1 = int((p1[1:][len_ - 2:len_].cpu() == lab[1:][len_ - 2:len_]).all())

            # 口径2：把标签词两个 token 对应的 logits 相加，取最大得到类别
            l2 = dict_ver["-".join([str(i) for i in lab[1:][len_ - 2:len_].tolist()])]
            p20 = p[len_ - 2 + 1]
            p21 = p[len_ - 1 + 1]
            p_l = int(torch.argmax(p20[k0] + p21[k1]))
            r2 = int(l2 == p_l)

            if r1 == 1 and r2 == 0:  # 口径1对但口径2错，定位到这种情况打印标记便于分析
                print("")

            right1 += r1
            right2 += r2


def cal_acc1(all_pre, all_lab, all_len):
    """
    评估口径1：逐 token 精确匹配。
    只有两个 [MASK] 位置预测出的 token id 与真实标签词 token id 完全一致才算对。
    """
    d_lens = sum([len(i) for i in all_pre])  # 样本总数
    right = 0

    for pres, labs, lens in zip(all_pre, all_lab, all_len):
        for p, lab, len_ in zip(pres, labs, lens):
            p = torch.argmax(p, dim=-1)  # 每个位置取预测 token
            # 取 [MASK] 位置（len_-2 到 len_）与真实标签对比，全部相等才算对
            right += int((p[1:][len_ - 2:len_].cpu() == lab[1:][len_ - 2:len_]).all())

    acc = right / d_lens * 100
    return acc


def cal_acc2(all_pre, all_label, all_len):
    """
    评估口径2：加和式投票。
    标签词被切成两个 token，把这两个 token 各自的 logits 相加，取和最大的那个标签词对应的类别。
    这种口径比「每个 token 都必须逐位命中」更宽容，通常准确率更高。
    """
    right = 0
    d_lens = sum([len(i) for i in all_pre])

    # 提前把 dict_ver 的 key 拆成两个 token id 张量，加速计算
    k0, k1 = zip(*[i.split("-") for i in list(dict_ver.keys())])
    k0, k1 = torch.tensor([int(i) for i in k0]), torch.tensor([int(i) for i in k1])

    for pres, labs, lens in zip(all_pre, all_label, all_len):
        for p, lab, len_ in zip(pres, labs, lens):
            # 真实标签词两个 token id 拼接起来就是 dict_ver 的 key -> 得到真实类别
            lab = dict_ver["-".join([str(i) for i in lab[1:][len_ - 2:len_].tolist()])]

            # 取两个 [MASK] 位置对应的 logits 向量
            p0 = p[len_ - 2 + 1]
            p1 = p[len_ - 1 + 1]

            # 把「每个标签词两个 token 位置」的 logits 相加，取最大对应的类别
            p_l = torch.argmax(p0[k0] + p1[k1])

            right += int(lab == int(p_l))

    acc = right / d_lens * 100
    return acc


if __name__ == "__main__":

    # dict_ver：手动预先算好的「类别标签词 -> 两个 token id」映射表。
    #   例如 "6568-5307" 表示该类别词在 RoBERTa 词表里被切成 token id 6568 和 5307。
    #   这是方案二（加和式投票）评估时用来复原类别用的。token id 需用 tokenizer 实际编码得到。
    dict_ver = {"6568-5307": 0,
                "2791-772": 1,
                "5500-4873": 2,
                "3136-5509": 3,
                "4906-2825": 4,
                "4852-833": 5,
                "3198-3124": 6,
                "860-5509": 7,
                "3952-2767": 8,
                "2031-727": 9, }

    # 读取训练集（抽查 3000 条）和测试集（抽查 400 条）
    train_texts, train_labels = read_data("train", 3000)
    dev_texts, dev_labels = read_data("test", 400)

    # 读取「类别索引 -> 标签词文本」的文件（如 0 -> 财经）
    with open(os.path.join("data", "index_2_label.txt"), encoding="utf-8") as f:
        index_2_label = f.read().split("\n")

    # 校验文本和标签数量一致
    assert len(train_texts) == len(train_labels)
    assert len(dev_texts) == len(dev_labels)

    # ---- 超参数配置 ----
    max_len = 50          # （预留，本脚本未严格使用）
    batch_size = 25       # 训练 batch 大小
    epoch = 250           # 训练轮数
    class_num = 21128     # 词表大小（这里仅作记录，本脚本未使用）
    lr = 0.001            # 学习率
    model_name = "hfl/chinese-roberta-wwm-ext"  # 预训练模型名

    # 选择设备：有 GPU 用 GPU，否则用 CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 构建数据集和 DataLoader
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, index_2_label)
    # collate_fn 指向自定义的 pro_batch_data，负责在读取时完成模板拼接
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=train_dataset.pro_batch_data)

    dev_dataset = TextDataset(dev_texts, dev_labels, tokenizer, index_2_label)
    dev_dataloader = DataLoader(dev_dataset, 10, shuffle=False, collate_fn=dev_dataset.pro_batch_data)

    # 初始化模型和优化器
    model = BertClassifier(model_name).to(device)
    optim = torch.optim.AdamW(model.parameters())

    # ---- 训练循环 ----
    for e in range(epoch):
        model.train()
        for batch_index, (batch_x, batch_label, batch_len) in enumerate(train_dataloader):
            batch_x = batch_x.to(device)
            batch_label = batch_label.to(device)
            # 前向计算损失
            loss = model.forward(batch_x, batch_len, batch_label)
            # 反向传播 + 参数更新
            loss.backward()
            optim.step()
            optim.zero_grad()  # 清空梯度，防止累加
            if batch_index % 20 == 0:
                print(f"loss:{loss:.2f}")
            break  # 注意：每个 epoch 只训练 1 个 batch（作者为了快速演示/少样本实验）

        # ---- 验证 ----
        model.eval()  # 切到评估模式（关闭 dropout 等）
        right = 0

        all_pre = []  # 收集每个 batch 的预测 logits
        all_lab = []  # 收集对应的标签
        all_len = []  # 收集对应的真实长度
        for t_batch_index, (t_batch_x, t_batch_label, t_batch_len) in enumerate(dev_dataloader):
            t_batch_x = t_batch_x.to(device)
            # t_batch_label = t_batch_label.to(device)
            pre = model.forward(t_batch_x, t_batch_len)  # 无 label -> 返回 logits

            all_pre.append(pre)
            all_lab.append(t_batch_label)
            all_len.append(t_batch_len)
            if t_batch_index == 10:  # 只验证前 11 个 batch，加快演示
                break

        # 注意：下面这行被注释掉了，而后面 if 又用到了 acc1。
        # 【跑通提示】若需完整运行，请取消注释下一行，否则 `acc1` 未定义会报错。
        # acc1 = cal_acc1(all_pre, all_lab, all_len)
        acc2 = cal_acc2(all_pre, all_lab, all_len)

        if acc1 > acc2:  # 两种口径比较（口径1逐位匹配、口径2加和投票）
            cal__(all_pre, all_lab, all_len)

        print(f"acc1:{acc1:.3f}% ")
        print(f"acc2:{acc2:.3f}% ")