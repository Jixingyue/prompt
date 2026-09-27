# Prompt 文本分类项目说明

本项目用两种「提示学习（Prompt-based Learning）」的方法，基于中文 RoBERTa 预训练模型完成中文新闻文本的**少样本分类**任务。相比传统「文本 -> 类别」的微调方式，提示学习能更充分地利用预训练模型的知识，在标注数据较少的场景下表现更好。

> 面向人群：刚接触大模型 / 提示学习的新手。建议先看「核心思想」和「数据格式」，再对照代码注释阅读。

---

## 一、项目目录结构

```
prompt/
├── prompt_文本分类.py      # 方案一：硬提示模板 + MLM（完形填空）分类
├── ptuning_文本分类.py     # 方案二：P-tuning（可学习连续提示）分类
└── README.md              # 本文档
```

运行所需的数据和模型**不在本目录内**，请自行准备（详见「数据格式」与「如何跑通」）：

- 数据目录 `data/`（需包含 `train.txt`、`test.txt`、`index_2_label.txt`）
- 预训练模型 `hfl/chinese-roberta-wwm-ext`（联网自动下载，或本地缓存）

---

## 二、核心思想（用大白话解释）

### 分类任务
给定一条新闻文本，判断它属于哪个类别（分类共有 10 类：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐）。

### 传统做法 vs 提示学习

| | 传统微调 | 提示学习（本项目） |
|---|---|---|
| 输入 | 只给一句话 | 给一句「带空格的模板话术」 |
| 模型任务 | 直接猜类别 | 做「完形填空」，把空格填上 |
| 监督信号 | 类别标签 | 空格处应填的词 |
| 可训参数 | 通常是整个模型 | 很少（甚至只训输出头/prompt） |

### 方案一：硬提示模板（prompt_文本分类.py）

人为定死一句提示模板：

```
文章内容为：<原始文本> 类别是：[MASK][MASK]
```

- `[MASK]` 的个数 = 标签词被词表切分的 token 数（如「财经」→ `[6568, 5307]` 两个 token）。
- 训练目标是让模型在 `[MASK]` 位置**预测出正确标签词**的 token id。
- 只需微调 MLM 输出头附近参数，BERT 主体冻结。

### 方案二：P-tuning（ptuning_文本分类.py）

不再用写死的文字模板，而是在句子最前面插入**20 个可学习的连续向量**（`<prompt_0>` ~ `<prompt_19>`），其中两个固定位置放 `[MASK]` 用来做分类。

- BERT 主体**全部冻结**，只训练：① 这些 prompt 向量 ② 两个小的分类头。
- 损失 = MLM 损失（填对空格）+ 分类损失（`[CLS]` 直接分对类别），各占一半。
- 参数量极小，极度适合少样本。

---

## 三、数据格式

### 1. `data/train.txt`、`data/test.txt`

每行一条样本，用 **Tab 键（`\t`）** 分隔「文本」和「类别索引」：

```
中华女子学院：本科层次仅1专业招男生	3
两天价网站背后重重迷雾：做个网站究竟要多少钱	4
东5环海棠公社230-290平2居准现房98折优惠	1
```

类别索引为 `0~9`，与 `index_2_label.txt` 的行号一一对应。

### 2. `data/index_2_label.txt`

一行一个标签词，**行号即类别索引**（第 1 行是类别 0）：

```
财经
房产
股票
教育
科技
社会
时政
体育
游戏
娱乐
```

### 3. 关于代码里的 `dict_ver`

`dict_ver` 是作者预先算好的「标签词 -> 两个 token id」映射表，例如：

```python
dict_ver = {"6568-5307": 0, "2791-772": 1, ...}
```

含义：标签词「财经」被 RoBERTa 词表切成了 token id `6568` 和 `5307`（每个中文标签词通常切成 2 个 token）。

> ⚠️ 该映射**与 `hfl/chinese-roberta-wwm-ext` 词表绑定**。若更换预训练模型，需要重新用
> `tokenizer.encode(标签词, add_special_tokens=False)` 计算并更新这张表。

---

## 四、环境依赖

- Python 3.8+
- `torch`
- `transformers`

安装示例：

```bash
pip install torch transformers
```

> 首次运行会从 HuggingFace 下载 `hfl/chinese-roberta-wwm-ext` 模型，需联网；也可提前下载放到本地缓存。

---

## 五、如何跑通

1. **准备数据**：在本目录下新建 `data/` 文件夹，放入 `train.txt`、`test.txt`、`index_2_label.txt`（格式见上文）。

2. **运行方案一（硬提示模板）**：

   ```bash
   python prompt_文本分类.py
   ```

3. **运行方案二（P-tuning）**：

   ```bash
   python ptuning_文本分类.py
   ```

4. **观察输出**：训练过程中会打印 `loss`，每个 epoch 后打印 `acc1`（准确率）。

---

## 六、代码逻辑详解

### 方案一 `prompt_文本分类.py`

调用链（自上而下）：

```
read_data() ──► TextDataset.__getitem__() ──► DataLoader(collate_fn=pro_batch_data)
                                                        │
                                                        ▼
                                              构造「模板 + MASK」，产出 text_idx / label_idx
                                                        │
                                                        ▼
                                     BertClassifier(AutoModelForMaskedLM).forward()
                                                        │
                                              [MASK] 位置交叉熵 → loss
                                                        │
                                                        ▼
                                           cal_acc1 / cal_acc2 评估准确率
```

| 函数/类 | 作用 |
|---|---|
| `read_data()` | 读取 `data/{train/test}.txt`，可随机抽样 num 条 |
| `TextDataset.__getitem__()` | 取单条「文本 + 标签」，文本截断到 100 字 |
| `TextDataset.pro_batch_data()` | collate_fn：拼模板、转 token id、构造监督标签、padding |
| `BertClassifier` | 加载 `AutoModelForMaskedLM`，冻结非 `cls` 参数，只训输出头 |
| `BertClassifier.forward()` | 前向得到词表 logits，训练返回 loss、推理返回 logits |
| `cal_acc1()` | 评估：两个 `[MASK]` 位置预测 token 与真实标签**逐个完全一致**才算对 |
| `cal_acc2()` | 评估：两个位置 logits **相加投票**得到类别（更宽松，通常更高） |

### 方案二 `ptuning_文本分类.py`

与方案一的主要区别集中在「prompt 的表示」：

```
extend_tokenizer() ──► 注册 <prompt_i> 特殊 token（id >= vocab_size）
        │
TextDataset.generate_template() ──► 生成 <prompt> + [MASK] 前缀
        │
Prompt_embedding ──► 把 prompt 索引转成可训练 768 维向量
        │
BertClassifier.forward():
   1. 按 id 大小把 batch_x 拆成「prompt 部分 / 普通文本部分」
   2. prompt → 查 Prompt_embedding；普通 token → 查 BERT 词嵌入
   3. 拼回完整 embedding 送进 BERT
   4. cls1 得词表 logits（MLM 损失）、cls2 得类别 logits（分类损失）
   5. loss = 0.5*loss1 + 0.5*loss2
```

| 函数/类 | 作用 |
|---|---|
| `extend_tokenizer()` | 向词表注册 `<prompt_0>...`，得到的 id 会 `>= vocab_size` |
| `TextDataset.generate_template()` | 按 `mask_position` 生成 `<prompt>` 与 `[MASK]` 交替的前缀 |
| `TextDataset.pro_batch_data()` | 构造输入 + 两种标签（token 级、类别级） |
| `Prompt_embedding` | 可训练的 prompt 向量层（Embedding + Linear） |
| `BertClassifier` | 冻结 BERT，只训 prompt 向量 + 两个分类头 |
| `BertClassifier.forward()` | 拆分/替换 embedding → BERT → 双头输出 → 双损失 |
| `cal_acc1()` | 用 `mask_position` 定位固定 mask 位置，逐 token 匹配 |
| `cal_acc2()` | 旧评估逻辑（残留，主程序未打印结果） |

---

## 七、两个方案对比

| 维度 | 方案一（硬提示） | 方案二（P-tuning） |
|---|---|---|
| 提示形式 | 固定中文模板文字 | 可训练连续向量 |
| BERT 主体 | 冻结，仅微调 MLM 输出头 | 全部冻结 |
| 额外可训结构 | 无 | Prompt_embedding + 两个分类头 |
| 是否改词表 | 否 | 是（新增 `<prompt_i>`） |
| 可训参数量 | 较少 | 极少 |
| 灵活性 | 模板固定 | 提示可学习、更灵活 |

---

## 八、注意事项（新手常见坑）

1. **缺少 `data/` 目录会报错**：两个脚本都会读 `data/train.txt`、`data/test.txt`、`data/index_2_label.txt`，运行前务必准备好。

2. **每个 epoch 只训练 1 个 batch**：训练循环末尾有 `break`，这是作者为「少样本 / 快速演示」特意写的，属于有意设计，不是 bug。

3. **方案一里 `acc1` 被注释掉了**：
   ```python
   # acc1 = cal_acc1(all_pre, all_lab, all_len)   # 这行被注释了
   acc2 = cal_acc2(all_pre, all_lab, all_len)
   if acc1 > acc2:   # 但这里又用到了 acc1
   ```
   若直接运行会在 `if acc1 > acc2` 处报 `NameError`（变量未定义）。**跑通前请取消注释 `acc1 = cal_acc1(...)` 这一行**（详见文件内注释标记处）。

4. **`dict_ver` 与模型绑定**：换预训练模型后，需重新用 tokenizer 计算标签词的 token id 并更新该表。

5. **评估精度说明**：方案二主程序实际只用 `acc1`（固定 mask 位置逐 token 匹配），`cal_acc2` 是沿用方案一的旧逻辑、未被打印，可忽略。