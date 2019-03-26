[图解BERT（NLP中的迁移学习）](https://www.cnblogs.com/d0main/p/10165671.html)
-------------------------------------------------------------------

目录

*   [一、例子:句子分类](#一例子句子分类)
*   [二、模型架构](#二模型架构)
    *   [模型的输入](#模型的输入)
    *   [模型的输出](#模型的输出)
*   [三、与卷积网络并行](#三与卷积网络并行)
*   [四、嵌入表示的新时代](#四嵌入表示的新时代)
    *   [回顾一下词嵌入](#回顾一下词嵌入)
    *   [ELMo: 语境的重要性](#elmo-语境的重要性)
*   [五、ULM-FiT：搞懂NLP中的迁移学习](#五ulm-fit搞懂nlp中的迁移学习)
*   [六、Transformer：超越LSTM](#六transformer超越lstm)
*   [七、OpenAI　Transformer：为语言建模预训练一个Transformer解码器](#七openaitransformer为语言建模预训练一个transformer解码器)
*   [八、在下游任务中使用迁移学习](#八在下游任务中使用迁移学习)
*   [九、BERT：从解码器到编码器](#九bert从解码器到编码器)
    *   [MLM语言模型](#mlm语言模型)
    *   [两个句子的任务](#两个句子的任务)
    *   [解决特定任务的模型](#解决特定任务的模型)
    *   [用于特征提取的BERT](#用于特征提取的bert)
*   [十、把BERT牵出来遛一遛](#十把bert牵出来遛一遛)

_本文翻译自Jay Alammar的博客[The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/)_

　　2018年是机器学习模型处理文本(更准确地说，是自然语言处理或简称NLP)的转折点。我们对以何种方式捕捉潜在语义和关系的来表示单词和句子这一问题的理解也在迅速发展。此外，NLP社区中也有人分享了许多了非常强大的模型供你免费下载并在自己的模型和pipeline中使用(它被称为NLP的ImageNet moment，参考了多年前计算机视觉的快速发展)。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223210725699-1866158801.png)

　　其中一个最新的里程碑就是BERT的发布，这被人们看作是NLP新时代的开始。BERT是一个打破了许多基于语言的任务中的记录。在论文发布后不久，该模型的团队还开源了模型的代码，并提供了模型的下载版本，这些模型已经在大数据集上进行过了预训练。这是一个重大的进步，因为它使任何想要构建自然语言处理的机器学习模型的人都可以将这个强大的预训练好的模型作为一个随时可用的组件使用——从而节省了从头开始训练模型所需的时间、精力、知识和资源。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223210736044-1390036503.png)

使用BERT的两个步骤。第一步：下载预训练好的模型；第二步：在特定任务上进行微调

　　BERT也是基于最近在NLP社区中涌现的许多聪明的想法，包括 [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432) (by [Andrew Dai](https://twitter.com/iamandrewdai) 和 [Quoc Le](https://twitter.com/quocleix)), [ELMo](https://arxiv.org/abs/1802.05365) (by [Matthew Peters](https://twitter.com/mattthemathman) 和来自 [AI2](https://allenai.org/) and [UW CSE](https://www.engr.washington.edu/about/bldgs/cse)的研究人员), [ULMFiT](https://arxiv.org/abs/1801.06146) (by fast.ai 创始人 [Jeremy Howard](https://twitter.com/jeremyphoward) 和大牛 [Sebastian Ruder](https://twitter.com/seb_ruder)), [OpenAI transformer](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (by OpenAI 研究员[Radford](https://twitter.com/alecrad), [Narasimhan](https://twitter.com/karthik_r_n), [Salimans](https://twitter.com/timsalimans), and [Sutskever](https://twitter.com/ilyasut)), 以及Transformer ([Vaswani et al](https://arxiv.org/pdf/1706.03762.pdf))等.

　　要正确理BERT是什么，我们需要了解许多概念。倒不如先看看BERT有哪些用途。

一、例子:句子分类
---------

　　最直接的想法就是使用BERT进行单个文本的分类。这个模型看起来是这样的:

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223210745557-1442527569.png)

　　要训练这样的模型，您需要重点训练一个分类器，在训练阶段对BERT模型的改动非常小。这种训练过程称为微调（fine-tuning），其根源在于 [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432) 和ULMFiT。

　　对于不熟悉这个概念的人来说，由于我们讨论的是分类器，所以在这个任务中涉及到的是机器学习中的监督学习。这意味着我们需要一个标记好的数据集来训练这样的模型。以垃圾邮件分类为例，标记的数据集将是一个电子邮件消息列表和一个标签（标注“垃圾邮件”或“非垃圾邮件”）。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223210755015-1535261829.png)

类似的任务场景还有:

*   情感分析
    *   输入: 一条影评/商品评价。
    *   输出: 正面评价还是负面评价?
    *   数据集如: [SST](https://nlp.stanford.edu/sentiment/)
*   事实核查
    *   输入:一个句子。
    *   输出: 是不是一个断言
    *   更难的任务:
        *   输入: 一句断言。
        *   输出: 真的还是假的
    *   [Full Fact](https://fullfact.org/) 组织构建了一个自动事实核查工具。这个工具的流程中包含了一个分类器，这个分类器读取新闻文章来检测断言.
    *   视频: [Sentence embeddings for automated factchecking - Lev Konstantinovskiy](https://www.youtube.com/watch?v=ddf0lgPCoSo).

二、模型架构
------

　　现在您已经有了一个BERT的用例，接下来让我们进一步了解它是如何工作的。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223210806121-909876168.png)

Google在论文中提到了两个不同模型规模的BERT:

*   BERT BASE –和OpenAI Transformer模型的规模差不多，方便与其进行性能比较
*   BERT LARGE – 一个达到目前多个benchmark的SOTA的巨大的模型

　　BERT基本上就是一个训练好的Transformer编码器栈。关于Transformer的内容可以看看 [图解Transformer](https://www.cnblogs.com/d0main/p/10164192.html)这篇博文。Transformer是BERT的基础，下面我们也会涉及到这个概念。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223210814164-1158823623.png)

　　两种规模的BERT模型都有许多编码器层 (在论文中称为“Transformer块”) – BERT Base有12个这样的结构，BERT Large有24个。编码器中也有前馈网络 (BERT Base中的是768个隐层神经元，BERT Large中的是1024个隐层神经元)， 以及注意力层中使用了比Transformer那篇论文中更多的“头” （BERT Base有12个“头”，BERT Large中有16个）。

### 模型的输入

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223210829873-1193630360.png)

　　输入序列的第一个token是一个特殊的符号\[CLS\]，这里的CLS代表class。

　　就像Transformer的编码器一样，BERT以一串单词作为输入，这些单词不断地想编码器栈上层流动。每一层都要经过自注意力层和前馈网络，然后在将其交给下一个编码器。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223210839217-1438175708.png)

　　在体系结构方面，到目前为止，还是与Transformer是相同的（除了一些超参数之外）。接下来在输出端，我们会看到其和Transformer的不同之处。

### 模型的输出

　　每个位置对应地输出一个维度为_hidden_size_(BERT Base中为768)的向量。对于之前提到的句子分类的例子，我们只关注第一个位置的输出（也就是被我们用\[CLS\]符号代替的位置）。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223210850418-1049075366.png)

　　输出的这个向量现在可以用作我们选择的分类器的输入。论文利用一个单层神经网络作为分类器，就能取得较好的分类效果。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223210858320-1831484936.png)

　　如果你有更多的标签（例如，如果你是一个电子邮件服务提供商，你需要将电子邮件标记为“垃圾邮件”、“非垃圾邮件”、“社交”和“促销”等等），你只需调整分类器这部分的网络，使其具有更多的输出神经元，然后通过softmax。

三、与卷积网络并行
---------

　　对于有CV背景的人来说，这种向量传递应该让人想起像VGGNet这样的网络的卷积部分和网络结构最后的全连接层之间发生的事情。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211005356-267958720.png)

四、嵌入表示的新时代
----------

　　这些新的探索带来了文本编码方式的新转变。到目前为止，在绝大多数的NLP模型中，词嵌入一直是一个主要的文本表示方法。Word2Vec、Glove等方法已广泛应用于此类任务。下面先让我们回顾一下如何使用它们。

### 回顾一下词嵌入

　　为了要让机器学习模型能够处理单词，我们需要以数字的形式表示文本，以便模型在计算中使用。通过使用Word2Vec，我们可以用一个向量来代表单词，而这一向量还捕捉了一定的语义信息（如“斯德哥尔摩”和“瑞典”的关系相当于“开罗”与“埃及”的关系)以及语法信息，或基于语法的关系（例如，“had”和“has”的关系与“was”和“is”的关系是一样的）。

　　人们很快意识到，使用大量文本数据进行预训练学习词嵌入是一个好主意，而不是在小数据集上模型从零开始训练。你可以下载预训练的Word2Vec或GloVe。下面是GloVe训练得到的“stick”对应的向量表示(嵌入向量维度为200）。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211013557-1739777055.png)

　　由于维度很大，在后面的文章中会用下面这种形状代表向量：

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211025230-185993939.png)

### ELMo: 语境的重要性

　　如果我们是使用GloVe训练好的向量作为一个词，比如“stick”的表示，那么不管在什么上下文中，这个表示都是一样的。在一些研究中 ([Peters et. al., 2017](https://arxiv.org/abs/1705.00108), [McCann et. al., 2017](https://arxiv.org/abs/1708.00107), [Peters et. al., 2018 in the ELMo paper](https://arxiv.org/pdf/1802.05365.pdf) )，研究人员认为像“_stick_”这样的词其实有很多意思，具体是什么意思取决于在什么语境中用它。那么为什么不基于其上下文语境来学习一个词的嵌入表示呢？也就是即学习到这个词的上下文的语义，有学习到其他的语境信息。就此，语境化的词嵌入模型应运而生。

![](https://jalammar.github.io/images/elmo-embedding-robin-williams.png)

语境化的词嵌入模型能够基于一个单词的上下文的意思给出单词的向量表示\[RIP Robin Williams\](https://www.youtube.com/watch?v=OwwdgsN9wF8)

　　ELMo没有为每个单词使用固定的嵌入，而是在为每个单词分配嵌入之前查看整个句子。它使用针对特定任务的双向LSTM来创建这些嵌入。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211058188-52590855.png)

　　ELMo为在语境中进行预训练提供了重要的思路。ELMo LSTM能够在大数据集上进行训练，然后作为其他模型的一个部分处理其他的自然语言任务。

**ELMo的秘诀是什么?**

　　ELMo通过训练预测单词序列中的下一个单词来理解语言——这项任务被称为**语言建模**。这很方便，因为我们有的是大量的文本数据，这样的模型可以从这些数据中学习，而不需要额外的标签。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211110992-93116852.png)

ELMo进行预训练的一个步骤：给定输入“Let’s stick to”， 预测接下来一个词，这就是语言模型的任务。当模型在大语料上进行预训练，他就会学习其中的语言模式。它不太可能准确地直接地猜出这个例子中的下一个单词。更实际一点说，在“hang”这样的单词之后，它将为“out”这样的单词分配更高的概率(组成 “hang out”) 而不是给“camera”分配更高的概率。

　　我们可以看到每个LSTM时间步的隐状态从ELMo的“头部”后面探出来。这些向量会在预训练结束后的嵌入过程中会派上用场。

　　ELMo实际上更进一步，训练了一个双向的LSTM——这样它的语言模型不仅能预测下一个词，还有预测上一个词。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211118781-1736365390.png)

\[Great slides\](https://www.slideshare.net/shuntaroy/a-review-of-deep-contextualized-word-representations-peters-2018) on ELMo

　　ELMo通过将隐藏状态(和初始嵌入)以某种方式(拼接之后加权求和)组合在一起，提出了语境化的词嵌入。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211128658-535774510.png)

五、ULM-FiT：搞懂NLP中的迁移学习
---------------------

　　ULM-FiT引入了一些方法来有效地利用模型在预训练中学到的东西——不仅仅是嵌入，还有语境化的嵌入表示。ULM-FiT引入了一个语言模型和一套针对各种任务有效地对语言模型进行微调的流程。

　　NLP终于找到了一种方法，可以像CV那样进行迁移学习了。

六、Transformer：超越LSTM
--------------------

　　Transformer的论文和代码的发布，以及它在机器翻译等任务上取得的成果，开始使一些业内人士认为它是LSTM的替代品。Transformer比LSTM更能处理长期依赖。

　　Transformer的编码器-译码器结构使其成为机器翻译的理想模型。但是你会如何使用它来进行句子分类呢？你将如何针对其他特定任务对语言模型进行微调呢？

七、OpenAI　Transformer：为语言建模预训练一个Transformer解码器
---------------------------------------------

　　事实证明，我们可以不用一个完整的Transformer来迁移学习并进行微调。我们可以只用Transformer的解码器就可以了。解码器是一个很好的选择，因为它能屏蔽掉后来的词（当进行逐词生成翻译时，这是一个很有用的特性）。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211140024-643336045.png)

The OpenAI Transformer 是由Transformer的解码器栈组成的

　　这个模型堆叠了12个解码器层。由于在这种设计中没有编码器，因此这些解码器层也不会有Transformer原文中的那种编码器-解码器注意力子层。但是，仍然还是有自注意力层。

　　有了这种结构，我们可以继续在相同的语言建模任务上进行训练模型：使用大量(未标记的)文本来预测下一个单词。只是，把7000本书的文本扔给模型，让它学习！书籍非常适合这类任务，因为它允许模型学习相关联的信息，而当您使用tweet或文章进行训练时，您无法获得这些信息。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211148444-177639109.png)

The OpenAI Transformer现在已经准备好被训练成能够预测下一个单词了

八、在下游任务中使用迁移学习
--------------

　　既然OpenAI　Transformer已经经过了预训练，而且它的各个层也经过了调整，我们就可以开始在下游任务中使用它了。让我们先来看看句子分类（将邮件信息分为“垃圾邮件”或“非垃圾邮件”）:

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211158799-1438659720.png)

How to use a pre-trained OpenAI transformer to do sentence clasification

　　OpenAI的论文列出了许多用于处理不同类型任务输入的输入变换。下图显示了模型的结构和执行不同任务时的输入变换。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211210314-861763164.png)

九、BERT：从解码器到编码器
---------------

　　openAI Transformer为我们提供了一个基于Transformer的可微调的预训练的模型。但是把LSTM换成Transformer还是让有些东西丢失了。ELMo的语言模型是双向的，而openAI Transformer则只训练一个从左到右的语言模型。那么我们能否建立一个既能从左到右预测又能从右到左预测（同时受上、下文的制约)的基于Transformer的模型呢？

### MLM语言模型

　　_“我们将使用Transformer编码器”_，BERT说。

　　_“这太疯狂了”_，有人说，“_每个人都知道双向条件作用会让每个词在多层次的语境中间接地看到自己。_”

　　“_我们将使用掩码_”，BERT自信地说。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211219214-2033398179.png)

BERT遮罩住15%输入序列中15%的token，然后让模型预测这些遮罩住的位置是什么单词

　　找到合适的任务来训练Transformer的编码器栈是一个复杂的问题，BERT采用了早期文献(完形填空任务)中的“带掩码的语言模型”概念来解决这个问题。

　　除了屏蔽15%的输入，BERT还混入一些东西，以改进模型的微调方式。有时它会随机地将一个单词替换成另一个单词，并让模型预测该位置的正确单词。

### 两个句子的任务

　　如果你还记得OpenAI Transformer处理不同任务时所做的输入变换，你会注意到一些任务需要模型处理关于两个句子的信息（例如，一个句子是否是另一个句子的复述；再例如假设一个维基百科条目作为输入，一个关于这个条目的问题作为另一个输入，我们能回答这个问题吗？）

　　为了让BERT更好地处理多个句子之间的关系，预训练的过程还有一个额外的任务：给定两个句子（A和B）， B可能是接在A后面出现的句子吗？

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211229613-602551603.png)

BERT预训练的第二个任务是两个句子的分类任务。

### 解决特定任务的模型

　　BERT论文展示了BERT在不同任务上的应用。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211239650-626598805.png)

### 用于特征提取的BERT

　　微调的方法并不是使用BERT的唯一方法。就像ELMo一样，你也可以使用预训练好的BERT来创建语境化的词嵌入。然后，您可以将这些嵌入表示喂给现有的模型——论文中也提到，在NER这类任务中，这种用法的最终效果也没有比用微调的方法的结果差很多。

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211317303-1228266810.png)

　　哪种向量作为语境化嵌入的效果最好？我认为这取决于具体任务。论文比较了6中选择（与微调后的96.4分模型相比):

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223211304478-1913069689.png)

十、把BERT牵出来遛一遛
-------------

　　试用BERT的最好方法是通过在谷歌Colab上托管的[BERT FineTuning with Cloud TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb) notebook。如果你之前从未使用过Cloud TPU，那么这也是尝试它们的一个很好的开始，而且BERT代码也可以在TPU、CPU和GPU上工作。

　　下一步可以看看 [BERT代码实现](https://github.com/google-research/bert):

*   模型在 [modeling.py](https://github.com/google-research/bert/blob/master/modeling.py) (`class BertModel`)中定义，而且和原生的Transformer encoder非常相似。
*   [run_classifier.py](https://github.com/google-research/bert/blob/master/run_classifier.py) 是一个微调过程的例子。其中构造了一个分类层。如果你想构建自己的分类器，可以看看文件中的 `create_model()`方法。
*   一些预训练模型可供下载。其中包括了BERT Base和 BERT Large，以及在中文、英文等102种语言的维基百科上预训练得到的模型。
*   BERT并不是直接处理单词，而是把 WordPieces作为token。 [tokenization.py](https://github.com/google-research/bert/blob/master/tokenization.py) 是能够将单词转换成wordPieces的脚本。

　　你也可以参考 [BERT的PyTorch实现](https://github.com/huggingface/pytorch-pretrained-BERT)。[AllenNLP](https://github.com/allenai/allennlp) 用这个代码让其他模型也能够用[BERT的嵌入表示](https://github.com/allenai/allennlp/pull/2067) 。
