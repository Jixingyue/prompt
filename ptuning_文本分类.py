# -*- coding: utf-8 -*-
"""
【方案二】基于 P-tuning（可学习连续提示）的少样本文本分类
====================================================================
整体思路：

1. 方案一（prompt_文本分类.py）用的是「写死的文字模板」，模板内容是固定的中文，
   模型只能靠这些固定文字去引导补全。

2. P-tuning 的思路更进一步：不再用固定文字做提示，而是在输入序列的最前面插入
   若干「可学习的连续向量」（soft prompt，即方案里的 <prompt_i>）。
   这些向量一开始是随机/可训练的，训练时才逐步学会「最合适的提示」。

3. 具体做法：
   - 向 tokenizer 注册一批新的特殊 token：<prompt_0> ... <prompt_19>
     （它们的 token id 会排在本词表之后，即 >= vocab_size）。
   - 这些 <prompt_i> 不用 BERT 原来的词向量，而是用一个独立的、可训练的
     Prompt_embedding 层来生成它们的向量。
   - 训练时，把普通 token（文本、[MASK] 等）用 BERT 词向量表示，
     把 prompt 位置替换成可训练向量，然后一起送进 BERT。
   - 同时冻结 BERT 全部参数，只训练：① prompt 向量 ② 两个额外的分类头
     （cls1 做词表级预测、cls2 做类别预测），参数量极小，特别适合少样本。

4. 损失由两部分组成：
   - loss1：MLM 式损失，让 [MASK] 位置预测出正确标签词 token；
   - loss2：普通分类损失，用 [CLS] 向量预测类别；
   最终 loss = 0.5*loss1 + 0.5*loss2。

关键词：soft prompt、continuous prompt、P-tuning v1。
"""

import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import random
import copy  # 用于深拷贝输入，避免原地修改影响原张量
from transformers import BertTokenizer
from transformers import BertModel, AutoModelForMaskedLM, AutoModel


def read_data(train_or_test, num=None):
    """
    读取数据集文件（与方案一完全相同）。

    数据文件每行格式：  "原始文本\t类别索引"
    """
    with open(os.path.join("data", train_or_test + ".txt"), encoding="utf-8") as f:
        all_data = f.read().split("\n")  # 按换行拆行

    texts = []
    labels = []
    for data in all_data:
        if data:  # 跳过空行
            t, l = data.split("\t")  # \t 分隔文本和标签
            texts.append(t)
            labels.append(l)

    if num == None:
        return texts, labels
    else:
        sm_idx = random.sample(range(0, len(texts)), num)  # 随机抽样 num 条
        return [texts[i] for i in sm_idx], [labels[i] for i in sm_idx]


def extend_tokenizer(num_prompt_tokens, tokenizer):
    """
    向 tokenizer 中注册 num_prompt_tokens 个新的特殊 token：
        <prompt_0>, <prompt_1>, ... <prompt_{n-1}>
    这样 tokenizer.encode("<prompt_i>") 就能得到对应的 token id，
    且这些 id 会自动排在原词表之后（>= vocab_size），从而和普通 token 区分开。
    """
    prompt_tokens = []
    for i in range(0, num_prompt_tokens):
        token = '<prompt_{}>'.format(i)
        prompt_tokens.append(token)

    # add_special_tokens 会扩展词表，并返回新增 token 的数量
    tokenizer.add_special_tokens({"additional_special_tokens": prompt_tokens})


class TextDataset(Dataset):
    """数据集类：负责生成「prefix 模板 + 文本」的输入。

    与方案一最大的区别：
        - 模板不再是固定文字，而是由 <prompt_i> 和 [MASK] 交替拼成的可训练前缀。
    """

    def __init__(self, texts, labels, tokenizer, index_2_label, prompt_num, mask_position):
        self.texts = texts
        self.labels = labels

        self.tokenizer = tokenizer
        self.index_2_label = index_2_label  # 类别索引 -> 标签词文本
        self.prompt_num = prompt_num        # 一共使用多少个 prompt token
        self.mask_position = mask_position  # 元组，指定 [MASK] 位于 prefix 中的哪两个位置
        self.prefix = self.generate_template()  # 生成好的前缀模板

    def generate_template(self):
        """
        生成前缀模板。

        规则：
            prefix 总长度 = prompt_num + len(mask_position)。
            在 mask_position 里指定的位置放 [MASK]，
            其它位置依次放 <prompt_0>, <prompt_1>, ...

        例如 prompt_num=20、mask_position=(3,7) 时生成：
            <prompt_0><prompt_1><prompt_2>[MASK]<prompt_3><prompt_4><prompt_5>[MASK]<prompt_6>...
        共 20 个 <prompt_i> + 2 个 [MASK] = 22 个 token。
        """
        assert self.prompt_num >= 2  # 至少要能放下两个 mask

        prefix_template = []

        p_n = 0
        while p_n < self.prompt_num:
            # 当前已填入的长度如果是 mask 位置，就放 [MASK]（且不消耗 prompt 计数）
            if len(prefix_template) in self.mask_position:
                prefix_template.append("[MASK]")
            else:
                # 否则放入一个 prompt token，并递增编号
                prefix_template.append(f"<prompt_{p_n}>")
                p_n += 1

        # 下面是作者留下的另一种写法（已注释，逻辑等价，可忽略）
        # for i in range(self.prompt_num):
        #     if i in self.mask_position:
        #         prefix_template.append("[MASK]")
        #
        #     prefix_template = prefix_template + '<prompt_{}>'.format(i)

        return "".join(prefix_template)

    def __getitem__(self, index):
        # 文本只保留前 100 个字符
        text = self.texts[index][:100]
        label = int(self.labels[index])  # 标签 -> int 类别索引
        # 估算序列长度：文本长度 + prompt token 数 + 一些特殊 token 预留（用于后续算 max_len）
        t_len = len(text) + self.prompt_num + 3

        # 最终输入 = prefix 模板 + [SEP] + 原始文本
        text = self.prefix + "[SEP]" + text
        return text, label, t_len

    def __len__(self):
        return len(self.labels)

    def pro_batch_data(self, batch_data):
        """
        collate_fn：把 batch 内样本拼成模型输入，并构造两种监督标签。

        返回：
            batch_text_idx : 输入 token id（含 prompt token 和文本）
            batch_label_idx: token 级标签（label1），只在两个 [MASK] 位置放标签词 token，其余为 0
            batch_l        : 样本的真实类别索引（label2，用于分类损失）
            batch_text_len : 各样本去掉特殊 token 后的长度（评估定位用）
        """
        batch_text, batch_label, batch_len = zip(*batch_data)
        max_len = max(batch_len) + 3  # 本 batch 最大长度（预留少量空间）

        batch_text_idx = []
        batch_label_idx = []
        batch_l = []
        batch_text_len = []

        for text, label, len_ in zip(batch_text, batch_label, batch_len):
            # 标签词被切分后的 token id（例如「财经」-> [6568, 5307]）
            li = self.tokenizer.encode(self.index_2_label[label], add_special_tokens=False)

            # 文本 + prompt 一起 encode，会自动加 [CLS] 开头、[SEP] 结尾
            text_idx = self.tokenizer.encode(text, add_special_tokens=True)
            # 记录去掉 [CLS] 和两个 [SEP] 后的长度
            batch_text_len.append(len(text_idx) - 3)

            # 构造 token 级标签 label_idx：
            #   位置0= [CLS] 忽略
            #   之后 mask_position[0] 个位置 = 0（对应前面的 <prompt>，不参与损失）
            #   第 1 个 [MASK] 位置放标签词第 1 个 token
            #   中间夹着的 prompt 位置填 0
            #   第 2 个 [MASK] 位置放标签词第 2 个 token
            label_idx = [0] + [0] * self.mask_position[0] + [li[0]] + \
                [0] * (self.mask_position[1] - self.mask_position[0] - 1) + [li[1]]

            # padding 到 max_len
            text_idx += [0] * (max_len - len(text_idx))
            label_idx += [0] * (max_len - len(label_idx))
            batch_l.append(label)  # 真实类别，用于分类损失

            assert len(text_idx) == len(label_idx)

            batch_text_idx.append(text_idx)
            batch_label_idx.append(label_idx)

        return torch.tensor(batch_text_idx), torch.tensor(batch_label_idx), \
            torch.tensor(batch_l), batch_text_len


class Prompt_embedding(nn.Module):
    """可训练的 prompt 向量层。

    prompt 的每个特殊 token <prompt_i> 不会用 BERT 本身的词向量，而是用这里
    初始化的（可训练的）Embedding 来生成一个 768 维向量，再过一个线性层增强表达。

    prompt_num : prompt token 的个数（向量索引 0 ~ prompt_num-1）
    """

    def __init__(self, prompt_num):
        super(Prompt_embedding, self).__init__()
        # 每个 prompt token 对应一个 768 维向量（和 BERT 隐藏维度一致）
        self.embedding = nn.Embedding(prompt_num, 768)
        # self.embedding.weight.requires_grad = False
        # self.gru = nn.GRU(768, 256, batch_first=True)   # 作者尝试过的 GRU 增强（已注释）
        self.linear = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(768, 768),  # 对 prompt 向量做一次线性变换
            # nn.LeakyReLU()
        )

    def forward(self, x):
        # x 是 prompt token 的索引（0 ~ prompt_num-1）
        emb = self.embedding(x)
        # gru_out,_ = self.gru(emb.reshape(1,*emb.shape))
        x = self.linear(emb)  # 线性变换后输出
        # return x.reshape(*x.shape[1:])
        return x


class BertClassifier(nn.Module):
    """P-tuning 主模型。

    结构：
        - 冻结 BERT 全部参数
        - Prompt_embedding：把 <prompt_i> token 映射成可训练向量
        - cls1：Linear(768 -> vocab_size)，做词表级预测（用于 MLM 式损失）
        - cls2：Linear(768 -> num_class)，用 [CLS] 向量做类别预测（用于分类损失）
    """

    def __init__(self, prompt_num, model_name, num_class):
        super().__init__()
        # 可训练的 prompt 向量层
        self.pro_emb = Prompt_embedding(prompt_num)

        # 加载 BERT 主体（不需要 MLM 头，只用 encoder）
        self.bert = BertModel.from_pretrained(model_name)
        for name, param in self.bert.named_parameters():
            # 冻结 BERT 全部参数（注意：这里连输出相关层也一起冻结了）
            # if "cls" in name:
            #     param.requires_grad = True
            # else:
            #     param.requires_grad = False
            param.requires_grad = False

        # BERT 的词嵌入层：用于把普通 token id 转成向量
        self.bert_emb = self.bert.get_input_embeddings()
        self.vocab_size = self.bert.config.vocab_size  # 词表大小，用于区分 prompt token

        # 两个可训练的分类头
        self.cls1 = nn.Linear(768, self.vocab_size)  # 词表级预测（MLM）
        self.cls2 = nn.Linear(768, num_class)        # 类别预测

        # 两种损失：loss1 忽略 label=0 的位置（非 mask 位置）；loss2 是普通分类交叉熵
        self.loss_fun1 = nn.CrossEntropyLoss(ignore_index=0)
        self.loss_fun2 = nn.CrossEntropyLoss()

    def forward(self, batch_x, batch_len, label1=None, label2=None):
        """
        前向计算。

        参数:
            batch_x : 输入 token id（prompt token 的 id 会 >= vocab_size，文本 token 则 < vocab_size）
            label1  : token 级标签（训练时）；label2 : 真实类别（训练时）
                      二者均为 None 表示推理

        核心流程：
            1. 根据 id 大小，把 batch_x 拆成「prompt 部分」和「普通文本部分」；
            2. prompt 部分查可训练的 Prompt_embedding，普通部分查 BERT 词表； 
            3. 拼回完整 embedding 后送进 BERT； 
            4. 用两个分类头分别得到词表 logits 和类别 logits。
        """
        # 深拷贝一份，用于区分 prompt token 和普通 token，避免污染原输入
        batch_x_copy = copy.deepcopy(batch_x)

        # prompt token 的 id >= vocab_size，据此得到「哪些位置是 prompt」的布尔掩码
        prompt_position = batch_x_copy >= self.vocab_size

        # —— 拆分出 prompt token 的 id ——
        batch_x[batch_x < self.vocab_size] = 0      # 非 prompt 位置全部置 0
        prompt_idx = batch_x                        # 现在只剩 prompt token 的原始 id

        # —— 拆分出普通 token 的 id ——
        batch_x_copy[batch_x_copy >= self.vocab_size] = 0  # prompt 位置全部置 0
        bert_idx = batch_x_copy                    # 现在只剩普通 token 的 id

        # 查可训练 prompt 向量：
        #   prompt_idx>0 筛出真正的 prompt token 位置，减去 vocab_size 得到 0~prompt_num-1 的索引
        prompt_emb = self.pro_emb.forward(prompt_idx[prompt_idx > 0] - self.vocab_size)

        # 查 BERT 词向量（普通 token）
        bert_emb = self.bert_emb.forward(bert_idx)

        # 把 prompt 向量「塞回」它们原本所在的位置（其余位置仍用 BERT 词向量）
        bert_emb[prompt_position] = prompt_emb

        # 送进 BERT：返回 last_hidden_state（bert_out1）和 pooler_output（bert_out2，即 [CLS]）
        bert_out1, bert_out2 = self.bert(inputs_embeds=bert_emb, return_dict=False)
        pre1 = self.cls1(bert_out1)  # 词表 logits (batch, seq_len, vocab_size)
        pre2 = self.cls2(bert_out2)  # 类别 logits (batch, num_class)

        if label1 is not None and label2 is not None:
            # 训练：MLM 损失（只在 mask 位置） + 分类损失，各占一半
            loss1 = self.loss_fun1(pre1.reshape(-1, pre1.shape[-1]), label1.reshape(-1))
            loss2 = self.loss_fun2(pre2, label2)
            loss = loss1 * 0.5 + loss2 * 0.5
            return loss
        else:
            # 推理：返回词表 logits（评估用）
            return pre1
            # return torch.argmax(pre, dim=-1)


# ---------------------------------------------------------------------------
# 评估/调试辅助函数（注意引用全局 dict_ver、mask_position）
# ---------------------------------------------------------------------------

def cal__(all_pre, all_lab, all_len):
    """调试用：统计两种评估口径对/错，遇到 acc1 对而 acc2 错时打印标记（与方案一同款）。"""
    right1 = 0
    right2 = 0
    d_lens = sum([len(i) for i in all_pre])

    # 从 dict_ver 的 key 里拆出两个标签 token id，用于加和式投票
    k0, k1 = zip(*[i.split("-") for i in list(dict_ver.keys())])
    k0, k1 = torch.tensor([int(i) for i in k0]), torch.tensor([int(i) for i in k1])

    for pres, labs, lens in zip(all_pre, all_lab, all_len):
        for p, lab, len_ in zip(pres, labs, lens):
            p1 = torch.argmax(p, dim=-1)
            r1 = int((p1[1:][len_ - 2:len_].cpu() == lab[1:][len_ - 2:len_]).all())

            l2 = dict_ver["-".join([str(i) for i in lab[1:][len_ - 2:len_].tolist()])]
            p20 = p[len_ - 2 + 1]
            p21 = p[len_ - 1 + 1]
            p_l = int(torch.argmax(p20[k0] + p21[k1]))
            r2 = int(l2 == p_l)

            if r1 == 1 and r2 == 0:
                print("")

            right1 += r1
            right2 += r2


def cal_acc1(all_pre, all_lab, all_len):
    """
    评估口径1：逐 token 精确匹配。

    注意：本方案里 [MASK] 的位置是固定的 mask_position（而不是像方案一那样在末尾），
    所以这里直接用全局 mask_position 定位两个 mask 位置进行对比。
    """
    d_lens = sum([len(i) for i in all_pre])  # 样本总数
    right = 0

    for pres, labs, lens in zip(all_pre, all_lab, all_len):
        for p, lab, len_ in zip(pres, labs, lens):
            p = torch.argmax(p, dim=-1)  # 每个位置取预测 token id
            # p[1:] 去掉开头 [CLS]，再取 mask_position 的两个位置，与真实标签词 token 对比
            # （lab[1:] 同理去掉 CLS，才能和 mask_position 对齐）
            right += int((p[1:][[mask_position[0], mask_position[1]]].cpu() ==
                          lab[1:][[mask_position[0], mask_position[1]]]).all())

    acc = right / d_lens * 100
    return acc


def cal_acc2(all_pre, all_label, all_len):
    """
    评估口径2：加和式投票。

    注意：本函数仍沿用方案一「mask 在末尾（len_-2 到 len_）」的定位方式，
    与 P-tuning 方案里「mask 在固定 mask_position」并不一致，属于残留代码。
    主程序里 acc2 的打印已被注释，实际只用 acc1 作为指标。
    """
    right = 0
    d_lens = sum([len(i) for i in all_pre])

    k0, k1 = zip(*[i.split("-") for i in list(dict_ver.keys())])
    k0, k1 = torch.tensor([int(i) for i in k0]), torch.tensor([int(i) for i in k1])

    for pres, labs, lens in zip(all_pre, all_label, all_len):
        for p, lab, len_ in zip(pres, labs, lens):
            lab = dict_ver["-".join([str(i) for i in lab[1:][len_ - 2:len_].tolist()])]

            p0 = p[len_ - 2 + 1]
            p1 = p[len_ - 1 + 1]

            p_l = torch.argmax(p0[k0] + p1[k1])

            right += int(lab == int(p_l))

    acc = right / d_lens * 100
    return acc


if __name__ == "__main__":

    # dict_ver：手动预先算好的「类别标签词 -> 两个 token id」映射表（评估用）
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

    # 读取数据集（训练 3000 条，测试 300 条）
    train_texts, train_labels = read_data("train", 3000)
    dev_texts, dev_labels = read_data("test", 300)

    # 读取「类别索引 -> 标签词文本」
    with open(os.path.join("data", "index_2_label.txt"), encoding="utf-8") as f:
        index_2_label = f.read().split("\n")

    # 校验样本数量一致
    assert len(train_texts) == len(train_labels)
    assert len(dev_texts) == len(dev_labels)

    # ---- 超参数配置 ----
    max_len = 50
    batch_size = 20
    epoch = 250
    class_num = 21128       # 词表大小
    lr = 0.0001             # 学习率（比方案一小，因为这里训练的只是少量 prompt 参数）
    prompt_num = 20         # prompt token 数量
    mask_position = (3, 7)  # 两个 [MASK] 在 prefix 中的位置
    model_name = "hfl/chinese-roberta-wwm-ext"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    extend_tokenizer(prompt_num, tokenizer)  # 关键：向词表注册 <prompt_i> 特殊 token

    # 构建数据集和 DataLoader
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, index_2_label, prompt_num, mask_position)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=train_dataset.pro_batch_data)

    dev_dataset = TextDataset(dev_texts, dev_labels, tokenizer, index_2_label, prompt_num, mask_position)
    dev_dataloader = DataLoader(dev_dataset, 2, shuffle=False, collate_fn=dev_dataset.pro_batch_data)

    # 初始化模型与优化器（只有 prompt 向量 + 分类头可训练，参数量极小）
    model = BertClassifier(prompt_num, model_name, len(index_2_label)).to(device)
    optim = torch.optim.AdamW(model.parameters())

    # ---- 训练循环 ----
    for e in range(epoch):
        model.train()
        for batch_index, (batch_x, batch_label, batch_l, batch_len) in enumerate(train_dataloader):
            batch_x = batch_x.to(device)
            batch_label = batch_label.to(device)   # token 级标签 label1
            batch_l = batch_l.to(device)           # 类别标签 label2
            loss = model.forward(batch_x, batch_len, batch_label, batch_l)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if batch_index % 20 == 0:
                print(f"loss:{loss:.2f}")

            break  # 演示用：每个 epoch 只训练 1 个 batch

        # ---- 验证 ----
        model.eval()
        right = 0

        all_pre = []
        all_lab = []
        all_len = []
        for t_batch_index, (t_batch_x, t_batch_label, t_batch_label2, t_batch_len) in enumerate(dev_dataloader):
            t_batch_x = t_batch_x.to(device)
            # t_batch_label = t_batch_label.to(device)
            pre = model.forward(t_batch_x, t_batch_len)  # 推理：返回词表 logits

            all_pre.append(pre)
            all_lab.append(t_batch_label)  # 用 token 级标签做评估
            all_len.append(t_batch_len)
            # if t_batch_index == 10:
            #     break

        acc1 = cal_acc1(all_pre, all_lab, all_len)  # 主要指标：固定 mask 位置逐 token 匹配
        acc2 = cal_acc2(all_pre, all_lab, all_len)  # 残留的旧评估口径（未打印，可忽略）

        # if acc1 > acc2:
        #     cal__(all_pre, all_lab, all_len)

        print(f"acc1:{acc1:.3f}% ")
        # print(f"acc2:{acc2:.3f}% ")