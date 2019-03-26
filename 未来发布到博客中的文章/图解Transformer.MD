> 原文标题：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
> 
> 原文链接：[https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
> 
> 论文地址：[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

前言
--

      Attention这种机制最开始应用于机器翻译的任务中，并且取得了巨大的成就，因而在最近的深度学习模型中受到了大量的关注。在在这个基础上，我们提出一种完全基于Attention机制来加速深度学习训练过程的算法模型-Transformer。事实证明Transformer结构在特定任务上已经优于了谷歌的神经网络机器翻译模型。但是，Transformer最大的优势在于其在并行化处理上做出的贡献。谷歌也在利用Transformer的并行化方式来营销自己的云TPU。所以，现在让我们一步一步剖析Transformer的神秘面纱，让我看看他是怎么一步一步训练的。

     Transformer在Goole的一篇论文[Attention is All You Need](https://arxiv.org/abs/1706.03762)被提出，为了方便实现调用Transformer Google还开源了一个第三方库，基于TensorFlow的[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)，一个NLP的社区研究者贡献了一个Torch版本的支持：[guide annotating the paper with PyTorch implementation](http://nlp.seas.harvard.edu/2018/04/03/attention.html)。这里，我想用一些方便理解的方式来一步一步解释Transformer的训练过程，这样即便你没有很深的深度学习知识你也能大概明白其中的原理。

A High-Level Look
-----------------

       我们先把Transformer想象成一个黑匣子，在机器翻译的领域中，这个黑匣子的功能就是输入一种语言然后将它翻译成其他语言。如下图：

![](https://img-blog.csdnimg.cn/20181212114106874.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

掀起The Transformer的盖头，我们看到在这个黑匣子由2个部分组成，一个Encoders和一个Decoders。

![](https://img-blog.csdnimg.cn/20181212114429556.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

我们再对这个黑匣子进一步的剖析，发现每个Encoders中分别由6个Encoder组成（论文中是这样配置的）。而每个Decoders中同样也是由6个Decoder组成。

![](https://img-blog.csdnimg.cn/20181212114641957.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

对于Encoders中的每一个Encoder，他们结构都是相同的，但是并不会共享权值。每层Encoder有2个部分组成，如下图： 

![](https://img-blog.csdnimg.cn/20181212115014822.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

每个Encoder的输入首先会通过一个self-attention层，通过self-attention层帮助Endcoder在编码单词的过程中查看输入序列中的其他单词。如果你不清楚这里在说什么，不用着急，之后我们会详细介绍self-attention的。

Self-attention的输出会被传入一个全连接的前馈神经网络，每个encoder的前馈神经网络参数个数都是相同的，但是他们的作用是独立的。

每个Decoder也同样具有这样的层级结构，但是在这之间有一个Attention层，帮助Decoder专注于与输入句子中对应的那个单词（类似与[seq2seq models](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)的结构）

![](https://img-blog.csdnimg.cn/20181212153408451.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

Bringing The Tensors Into The Picture
-------------------------------------

      在上一节，我们介绍了Transformer的网络结构。现在我们以图示的方式来研究Transformer模型中各种张量/向量，观察从输入到输出的过程中这些数据在各个网络结构中的流动。

      首先还是NLP的常规做法，先做一个词嵌入：[什么是文本的词嵌入？](https://blog.csdn.net/qq_41664845/article/details/84313419)

![](https://img-blog.csdnimg.cn/20181212154347724.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

      我们将每个单词编码为一个512维度的向量，我们用上面这张简短的图形来表示这些向量。词嵌入的过程只发生在最底层的Encoder。但是对于所有的Encoder来说，你都可以按下图来理解。输入（一个向量的列表，每个向量的维度为512维，在最底层Encoder作用是词嵌入，其他层就是其前一层的output）。另外这个列表的大小和词向量维度的大小都是可以设置的超参数。一般情况下，它是我们训练数据集中最长的句子的长度。

![](https://img-blog.csdnimg.cn/20181212155212785.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

     上图其实介绍到了一个Transformer的关键点。你注意观察，在每个单词进入Self-Attention层后都会有一个对应的输出。Self-Attention层中的输入和输出是存在依赖关系的，而前馈层则没有依赖，所以在前馈层，我们可以用到并行化来提升速率。

    下面我用一个简短的句子作为例子，来一步一步推导transformer每个子层的数据流动过程。

Now We’re Encoding!
-------------------

       正如之前所说，Transformer中的每个Encoder接收一个512维度的向量的列表作为输入，然后将这些向量传递到‘self-attention’层，self-attention层产生一个等量512维向量列表，然后进入前馈神经网络，前馈神经网络的输出也为一个512维度的列表，然后将输出向上传递到下一个encoder。

![](https://img-blog.csdnimg.cn/20181212162052489.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

     如上图所示，每个位置的单词首先会经过一个self attention层，然后每个单词都通过一个独立的前馈神经网络（这些神经网络结构完全相同）。

Self-Attention at a High Level
------------------------------

      Self attention这个单词看起来好像每个人都知道是什么意思，但实质上他是算法领域中新出的概念，你可以通过阅读：[Attention is All You Need](https://arxiv.org/abs/1706.03762) 来理解self attention的原理。

      假设下面的句子就是我们需要翻译的输入句：

”`The animal didn't cross the street because it was too tired`”

      这句话中的"it"指的是什么？它指的是“`animal`”还是“`street`”？对于人来说，这其实是一个很简单的问题，但是对于一个算法来说，处理这个问题其实并不容易。self attention的出现就是为了解决这个问题，通过self attention，我们能将“it”与“`animal`”联系起来。

      当模型处理单词的时候，self attention层可以通过当前单词去查看其输入序列中的其他单词，以此来寻找编码这个单词更好的线索。

      如果你熟悉RNNs，那么你可以回想一下，RNN是怎么处理先前单词(向量）与当前单词(向量）的关系的？RNN是怎么计算他的hidden state的。self-attention正是transformer中设计的一种通过其上下文来理解当前词的一种办法。你会很容易发现...相较于RNNs，transformer具有更好的并行性。

![](https://img-blog.csdnimg.cn/20181212165538837.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

如上图，是我们第五层Encoder针对单词'it'的图示，可以发现，我们的Encoder在编码单词‘it’时，部分注意力机制集中在了‘animl’上，这部分的注意力会通过权值传递的方式影响到'it'的编码。

更多细节可以查看 [Tensor2Tensor notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) 

Self-Attention in Detail
------------------------

    这一节我们先介绍如何用向量的方式来计算self attention，然后再来看看它是如何使用矩阵来实现的。

     计算self attention的第一步是从每个Encoder的输入向量上创建3个向量（在这个情况下，对每个单词做词嵌入）。所以，对于每个单词，我们创建一个Query向量，一个Key向量和一个Value向量。这些向量是通过词嵌入乘以我们训练过程中创建的3个训练矩阵而产生的。

     注意这些新向量的维度比嵌入向量小。我们知道嵌入向量的维度为512，而这里的新向量的维度只有64维。新向量并不是必须小一些，这是网络架构上的选择使得Multi-Headed Attention（大部分）的计算不变。

![](https://img-blog.csdnimg.cn/20181212175547373.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

我们将![X_{1}](https://private.codecogs.com/gif.latex?X_%7B1%7D)乘以![W^{Q}](https://private.codecogs.com/gif.latex?W%5E%7BQ%7D)的权重矩阵得到新向量![q_{1}](https://private.codecogs.com/gif.latex?q_%7B1%7D)，![q_{1}](https://private.codecogs.com/gif.latex?q_%7B1%7D)既是“query”的向量。同理，最终我们可以对输入句子的每个单词创建“query”，  
“key”，“value”的新向量表示形式。

对了..“query”，“key”，“value”是什么向量呢？有什么用呢？

这些向量的概念是很抽象，但是它确实有助于计算注意力。不过先不用纠结去理解它，后面的的内容，会帮助你理解的。

      计算self attention的第二步是计算得分。以上图为例，假设我们在计算第一个单词“thinking”的self attention。我们需要根据这个单词对输入句子的每个单词进行评分。当我们在某个位置编码单词时，分数决定了对输入句子的其他单词的关照程度。

       通过将query向量和key向量点击来对相应的单词打分。所以，如果我们处理开始位置的的self attention，则第一个分数为![q_{1}](https://private.codecogs.com/gif.latex?q_%7B1%7D)和![k_{1}](https://private.codecogs.com/gif.latex?k_%7B1%7D)的点积，第二个分数为![q_{2}](https://private.codecogs.com/gif.latex?q_%7B2%7D)和![k_{2}](https://private.codecogs.com/gif.latex?k_%7B2%7D)的点积。如下图

![](https://img-blog.csdnimg.cn/20181212180838739.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

第三步和第四步的计算，是将第二部的得分除以8（![\sqrt{d_{k}}](https://private.codecogs.com/gif.latex?%5Csqrt%7Bd_%7Bk%7D%7D)）（论文中使用key向量的维度是64维，其平方根=8，这样可以使得训练过程中具有更稳定的梯度。这个![\sqrt{d_{k}}](https://private.codecogs.com/gif.latex?%5Csqrt%7Bd_%7Bk%7D%7D)并不是唯一值，经验所得）。然后再将得到的输出通过softmax函数标准化，使得最后的列表和为1。

![](https://img-blog.csdnimg.cn/20181212181833842.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

这个softmax的分数决定了当前单词在每个句子中每个单词位置的表示程度。很明显，当前单词对应句子中此单词所在位置的softmax的分数最高，但是，有时候attention机制也能关注到此单词外的其他单词，这很有用。

第五步是将每个Value向量乘以softmax后的得分。这里实际上的意义在于保存对当前词的关注度不变的情况下，降低对不相关词的关注。

第六步是 累加加权值的向量。 这会在此位置产生self-attention层的输出（对于第一个单词）。

![](https://jalammar.github.io/images/t/self-attention-output.png)

总结self-attention的计算过程，（单词级别）就是得到一个我们可以放到前馈神经网络的矢量。 然而在实际的实现过程中，该计算会以矩阵的形式完成，以便更快地处理。下面我们来看看Self-Attention的矩阵计算方式。

Matrix Calculation of Self-Attention
------------------------------------

第一步是去计算Query，Key和Value矩阵。我们将词嵌入转化成矩阵X中，并将其乘以我们训练的权值矩阵（![W^{Q}](https://private.codecogs.com/gif.latex?W%5E%7BQ%7D),![W^{K}](https://private.codecogs.com/gif.latex?W%5E%7BK%7D),![W^{V}](https://private.codecogs.com/gif.latex?W%5E%7BV%7D)）

![](https://jalammar.github.io/images/t/self-attention-matrix-calculation.png)

X矩阵中的每一行对应于输入句子中的一个单词。 我们看到的X每一行的方框数实际上是词嵌入的维度，图中所示的和论文中是有差距的。X（图中的4个方框论文中为512个）和q / k / v向量（图中的3个方框论文中为64个）

最后，由于我们正在处理矩阵，我们可以在一个公式中浓缩前面步骤2到6来计算self attention层的输出。

![](https://img-blog.csdnimg.cn/20181212185829399.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

The Beast With Many Heads
-------------------------

本文通过使用“Multi-headed”的机制来进一步完善self attention层。“Multi-headed”主要通过下面2中方式改善了attention层的性能：

1\. 它拓展了模型关注不同位置的能力。在上面例子中可以看出，”`The animal didn't cross the street because it was too tired`”，我们的attention机制计算出“`it`”指代的为“`animal`”，这在对语言的理解过程中是很有用的。

2.它为attention层提供了多个“representation subspaces”。由下图可以看到，在self attention中，我们有多个个Query / Key / Value权重矩阵（Transformer使用8个attention heads）。这些集合中的每个矩阵都是随机初始化生成的。然后通过训练，用于将词嵌入（或者来自较低Encoder/Decoder的矢量）投影到不同的“representation subspaces（表示子空间）”中。

![](https://img-blog.csdnimg.cn/20181212191926825.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

    通过multi-headed attention，我们为每个“header”都独立维护一套Q/K/V的权值矩阵。然后我们还是如之前单词级别的计算过程一样处理这些数据。

    如果对上面的例子做同样的self attention计算，而因为我们有8头attention，所以我们会在八个时间点去计算这些不同的权值矩阵，但最后结束时，我们会得到8个不同的![Z](https://private.codecogs.com/gif.latex?Z)矩阵。如下图：

![](https://img-blog.csdnimg.cn/20181212192729754.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

    瞧瞧，这会给我们后续工作造成什么问题？

    我们知道在self-attention后面紧跟着的是前馈神经网络，而前馈神经网络接受的是单个矩阵向量，而不是8个矩阵。所以我们需要一种办法，把这8个矩阵压缩成一个矩阵。

    我们怎么做？ 

   我们将这8个矩阵连接在一起然后再与一个矩阵![W^{O}](https://private.codecogs.com/gif.latex?W%5E%7BO%7D)相乘。步骤如下图所示：

![](https://img-blog.csdnimg.cn/20181212193517381.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

这样multi-headed self attention的全部内容就介绍完了。之前可能都是一些过程的图解，现在我将这些过程连接在一起，用一个整体的框图来表示一下计算的过程，希望可以加深理解。

![](https://img-blog.csdnimg.cn/20181212193647230.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNjY0ODQ1,size_16,color_FFFFFF,t_70)

      现在我们已经触及了attention的header，让我们重新审视我们之前的例子，看看例句中的“it”这个单词在不同的attention header情况下会有怎样不同的关注点。

![](https://jalammar.github.io/images/t/transformer_self-attention_visualization_2.png)

      如图：当我们对“it”这个词进行编码时，一个注意力的焦点主要集中在“`animal`”上，而另一个注意力集中在“`tired`” 

但是，如果我们将所有注意力添加到图片中，那么事情可能更难理解：

![](https://jalammar.github.io/images/t/transformer_self-attention_visualization_3.png)

Representing The Order of The Sequence Using Positional Encoding
----------------------------------------------------------------

\# 使用位置编码表示序列的顺序
----------------

我们可能忽略了去介绍一个重要的内容，就是怎么考虑输入序列中单词顺序的方法。

为了解决这个问题，transformer为每个输入单词的词嵌入上添加了一个新向量-位置向量。

为了解决这个问题，变换器为每个输入嵌入添加了一个向量。这些位置编码向量有固定的生成方式，所以获取他们是很方便的，但是这些信息确是很有用的，他们能捕捉大奥每个单词的位置，或者序列中不同单词之间的距离。将这些信息也添加到词嵌入中，然后与Q/K/V向量点击，获得的attention就有了距离的信息了。

![](https://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png)

为了让模型捕捉到单词的顺序信息，我们添加位置编码向量信息（POSITIONAL ENCODING）-位置编码向量不需要训练，它有一个规则的产生方式。

如果我们的嵌入维度为4，那么实际上的位置编码就如下图所示：

![](https://jalammar.github.io/images/t/transformer_positional_encoding_example.png)  
那么生成位置向量需要遵循怎样的规则呢？

观察下面的图形，每一行都代表着对一个矢量的位置编码。因此第一行就是我们输入序列中第一个字的嵌入向量，每行都包含512个值，每个值介于1和-1之间。我们用颜色来表示1，-1之间的值，这样方便可视化的方式表现出来：

![](https://jalammar.github.io/images/t/transformer_positional_encoding_large_example.png)

这是一个20个字（行）的（512）列位置编码示例。你会发现它咋中心位置被分为了2半，这是因为左半部分的值是一由一个正弦函数生成的，而右半部分是由另一个函数（余弦）生成。然后将它们连接起来形成每个位置编码矢量。  
 

位置编码的公式在论文（3.5节）中有描述。你也可以在中查看用于生成位置编码的代码[`get_timing_signal_1d()`](https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py)。这不是位置编码的唯一可能方法。然而，它具有能够扩展到看不见的序列长度的优点（例如，如果我们训练的模型被要求翻译的句子比我们训练集中的任何句子都长）。

The Residuals
-------------

这一节我想介绍的是encoder过程中的每个self-attention层的左右连接情况，我们称这个为：[layer-normalization](https://arxiv.org/abs/1607.06450) 步骤。如下图所示：

![](https://jalammar.github.io/images/t/transformer_resideual_layer_norm.png)

在进一步探索其内部计算方式，我们可以将上面图层可视化为下图：

![](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png)

 Decoder的子层也是同样的，如果我们想做堆叠了2个Encoder和2个Decoder的Transformer，那么它可视化就会如下图所示：

![](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)

The Decoder Side
----------------

     我们已经基本介绍完了Encoder的大多数概念，我们基本上也可以预知Decoder是怎么工作的。现在我们来仔细探讨下Decoder的数据计算原理，

     当序列输入时，Encoder开始工作，最后在其顶层的Encoder输出矢量组成的列表，然后我们将其转化为一组attention的集合（K,V）。（K,V）将带入每个Decoder的“encoder-decoder attention”层中去计算（这样有助于decoder捕获输入序列的位置信息）

![](https://jalammar.github.io/images/t/transformer_decoding_1.gif)  
完成encoder阶段后，我们开始decoder阶段，decoder阶段中的每个步骤输出来自输出序列的元素（在这种情况下为英语翻译句子）。

上面实际上已经是应用的阶段了，那我们训练阶段是如何的呢？

我们以下图的步骤进行训练，直到输出一个特殊的符号<end of sentence>，表示已经完成了。 The output of each step is fed to the bottom decoder in the next time step, and the decoders bubble up their decoding results just like the encoders did. 对于Decoder，和Encoder一样，我们在每个Decoder的输入做词嵌入并添加上表示每个字位置的位置编码

![](https://jalammar.github.io/images/t/transformer_decoding_2.gif)

Decoder中的self attention与Encoder的self attention略有不同：

       在Decoder中，self attention只关注输出序列中的较早的位置。这是在self attention计算中的softmax步骤之前屏蔽了特征位置（设置为 -inf）来完成的。

     “Encoder-Decoder Attention”层的工作方式与"Multi-Headed Self-Attention"一样，只是它从下面的层创建其Query矩阵，并在Encoder堆栈的输出中获取Key和Value的矩阵。

The Final Linear and Softmax Layer
----------------------------------

         Decoder的输出是浮点数的向量列表。我们是如何将其变成一个单词的呢？这就是最终的线性层和softmax层所做的工作。

线性层是一个简单的全连接神经网络，它是由Decoder堆栈产生的向量投影到一个更大，更大的向量中，称为对数向量

      假设实验中我们的模型从训练数据集上总共学习到1万个英语单词（“Output Vocabulary”）。这对应的Logits矢量也有1万个长度-每一段表示了一个唯一单词的得分。在线性层之后是一个softmax层，softmax将这些分数转换为概率。选取概率最高的索引，然后通过这个索引找到对应的单词作为输出。

![](https://jalammar.github.io/images/t/transformer_decoder_output_softmax.png)

 上图是从Decoder的输出开始到最终softmax的输出。一步一步的图解。

Recap Of Training
-----------------

现在我们已经讲解了transformer的训练全过程了，让我们回顾一下。

在训练期间，未经训练的模型将通过如上的流程一步一步计算的。而且因为我们是在对标记的训练数据集进行训练（机器翻译可以看做双语平行语聊），那么我们可以将模型输出与实际的正确答案相比较，来进行反向传播。

为了更好的理解这部分内容，我们假设我们输出的词汇只有（“a”，“am”，“i”，“thanks”，“student”和“<eos>”（“句末”的缩写））

![](https://jalammar.github.io/images/t/vocabulary.png)

在我们开始训练之前，我们模型的输出词汇是在预处理阶段创建的。

一旦我们定义了输出的词汇表，那么我们就可以使用相同宽度的向量来表示词汇表中的每个单词。称为one-hot编码。例如，我们可以使用下面向量来表示单词“am”：

![](https://jalammar.github.io/images/t/one-hot-vocabulary-example.png)

                                                           示例：我们的输出词汇表的one-hot编码

下一节我们再讨论一下模型的损失函数，我们优化的指标，引导一个训练有素且令人惊讶的精确模型。

The Loss Function
-----------------

假设我们正在训练一个模型，比如将“merci”翻译成“谢谢”。这意味着我们希望模型计算后的输出为“谢谢”，但由于这种模式还没有接受过训练，所以这种情况不太可能发生。

![](https://jalammar.github.io/images/t/transformer_logits_output_and_label.png)

  
这是因为模型的参数（权重）都是随机初始化的，因此（未经训练的）模型对每个单词产生的概率分布是具有无限可能的，但是我们可以通过其余实际我们期望的输出进行比较，然后利用反向传播调整所有模型的权重，使得输出更接近所需的输出。

那么如何比较算法预测值与真实期望值呢？

实际上，我们对其做一个简单的减法即可。你也可以了解[交叉熵](https://colah.github.io/posts/2015-09-Visual-Information/)和[Kullback-Leibler散度](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)来掌握这种差值的判断方式。

但是，需要注意的是，这只是一个很简单的demo，真实情况下，我们需要输出一个更长的句子，例如。输入：“je suis étudiant”和预期输出：“I am a student”。这样的句子输入，意味着我们的模型能够连续的输出概率分布。其中：

*   每个概率分布由宽度为vocab\_size的向量表示（在我们的示例中vocab\_size为6，但实际上可能为3,000或10,000维度）
*   第一概率分布在与单词“i”相关联的单元处具有最高概率
*   第二概率分布在与单词“am”相关联的单元格中具有最高概率
*   依此类推，直到第五个输出分布表示' `<end of sentence>`'符号，意味着预测结束。

![](https://jalammar.github.io/images/t/output_target_probability_distributions.png)  
上图为：输入：“je suis étudiant”和预期输出：“I am a student”的期望预测概率分布情况。

在算法模型中，虽然不能达到期望的情况，但是我们需要在训练了足够长时间之后，我们的算法模型能够有如下图所示的概率分布情况：

![](https://jalammar.github.io/images/t/output_trained_model_probability_distributions.png)

    现在，因为模型一次生成一个输出，我们可以理解为这个模型从该概率分布（softmax）矢量中选择了具有最高概率的单词并丢弃了其余的单词。            

   现在，因为模型一次生成一个输出，我们可以假设模型从该概率分布中选择具有最高概率的单词并丢弃其余的单词。

这里有2个方法：一个是贪婪算法（greedy decoding），一个是波束搜索（beam search）。波束搜索是一个优化提升的技术，可以尝试去了解一下，这里不做更多解释。

Go Forth And Transform
----------------------

如果你想更深入了解，可以尝试下面几个步骤:

*   阅读 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 这篇论文, the Transformer 的博客 ([Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)) 和 [Tensor2Tensor announcement](https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html)公告
*   观看 [Łukasz Kaiser’s talk](https://www.youtube.com/watch?v=rBCqOTEfxvg) 这个视频，了解更多细节。
*   试着通过 [Jupyter Notebook provided as part of the Tensor2Tensor repo](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) 以编程的角度学习
*   探索 [Tensor2Tensor repo](https://github.com/tensorflow/tensor2tensor).

后续工作：

*   [Depthwise Separable Convolutions for Neural Machine Translation](https://arxiv.org/abs/1706.03059)
*   [One Model To Learn Them All](https://arxiv.org/abs/1706.05137)
*   [Discrete Autoencoders for Sequence Models](https://arxiv.org/abs/1801.09797)
*   [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198)
*   [Image Transformer](https://arxiv.org/abs/1802.05751)
*   [Training Tips for the Transformer Model](https://arxiv.org/abs/1804.00247)
*   [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
*   [Fast Decoding in Sequence Models using Discrete Latent Variables](https://arxiv.org/abs/1803.03382)
*   [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)

