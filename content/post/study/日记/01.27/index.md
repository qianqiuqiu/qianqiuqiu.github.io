---
title: "转码学习day7"
date: 2026-01-27
categories: 
  - study
tag: 
  - study
math: true 
---

今天刷三题力扣，然后开始做项目，顺便温习一下MHA代码，尽可能可以默写。

lc：24、25、543

默写还是差了一点，很多时候传参记不清楚。

尝试了调用hugging face的api和学习本地调用下载好的大模型推理

`tokenizer = AutoTokenizer.from_pretrained(model_dir)`
`model = AutoModelForCausalLM.from_pretrained(model_dir)`,之前做minimind的部署预训练和SFT的时候，没有用huggingface，而是直接git下来，调整了一下数据集预训练。

pipeline直接把之前的推理代码封装好了，有点太方便了。

直接输出的话其实效果很糟糕，小模型复读情况严重。

得多加一点参数限制，能稍微看起来不那么口吃。

*max_length*=50,*num_return_sequences*=1,*truncation*=True,temperature*=0.7,*top_p*=0.9,*top_k*=50,*clean_up_tokenization_spaces*=True
输出中文时一般行业通用规则是后处理去掉字之间空格，我猜是英文一个单词中间就一个空格，如果模型要兼容象形和楔形，都采用一个token一个空格，然后针对不需要空格的语言在输出的时候再去掉空格即可。

temperature是在logits归一化之前放大差距，一般需要大于0.5，无限接近0是没有意义的。

top-p关注的是累计概率，只允许在“累计概率 ≥ p 的最小 token 集合”里采样。1-3，3个token，

p1
p1 + p2
p1 + p2 + p3
...，直到p1~px相加大于top-p，在这些token里采样。

top-p动态上限，top-k写多少就多少。

这俩可以一起用，取min（top-k，top-p-nums）。

中文的tokenizer比较特殊，比如说黄 ##黄是不一样的东西，有的时候可以单独一个字有含义，有的时候是词语一个含义，所以一个字分成两种分词结构来训练。

今天的内容还比较浅，主要是熟悉huggingface的使用。





