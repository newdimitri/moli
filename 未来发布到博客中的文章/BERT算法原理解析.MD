

1\. 前言
======

谷歌在2017年发表了一篇论文名字教Attention Is All You Need，**提出了一个只基于attention的结构来处理序列模型**相关的问题，比如机器翻译。传统的神经机器翻译大都是利用RNN或者CNN来作为encoder-decoder的模型基础，而谷歌最新的只基于Attention的Transformer模型摒弃了固有的定式，并没有用任何CNN或者RNN的结构。该模型可以高度并行地工作，所以在提升翻译性能的同时训练速度也特别快。


2\. Transformer模型结构
===================

Transformer的主体结构图：

![image](https://note.youdao.com/yws/api/personal/file/512B77E2B038424A8BD6FCB2A5AB91B6?method=download&shareKey=621bb170e4eb474aa027ea299b31413c)

2.1 Transformer的编码器解码器
----------------------

模型分为**编码器**和**解码器**两个部分。

*   编码器由6个相同的层堆叠在一起，每一层又有两个支层。第一个支层是一个多头的自注意机制，第二个支层是一个简单的全连接前馈网络。在两个支层外面都添加了一个residual的连接，然后进行了layer nomalization的操作。模型所有的支层以及embedding层的输出维度都是dmodeldmodeld_{model}。
*   解码器也是堆叠了六个相同的层。不过每层除了编码器中那两个支层，解码器还加入了第三个支层，如图中所示同样也用了residual以及layer normalization。具体的细节后面再讲。

2.2 输入层
-------

编码器和解码器的输入就是利用学习好的embeddings将tokens（一般应该是词或者字符）转化为d维向量。对解码器来说，利用线性变换以及softmax函数将解码的输出转化为一个预测下一个token的概率。

2.3 位置向量
--------

由于模型没有任何循环或者卷积，为了使用序列的顺序信息，需要将tokens的相对以及绝对位置信息注入到模型中去。论文在输入embeddings的基础上加了一个“位置编码”。位置编码和embeddings由同样的维度都是dmodeldmodeld_{model}所以两者可以直接相加。有很多位置编码的选择，既有学习到的也有固定不变的。

2.4 Attention模型
---------------

### 2.4.1 Scaled attention

论文中用的attention是基本的点乘的方式，就是多了一个所谓的scale。输入包括维度为dkdkd_k的queries以及keys，还有维度为dvdvd_v的values。计算query和所有keys的点乘，然后每个都除以dk−−√dk\\sqrt{d_k}（这个操作就是所谓的Scaled）。之后利用一个softmax函数来获取values的权重。  
实际操作中，attention函数是在一些列queries上同时进行的，将这些queries并在一起形成一个矩阵QQQ同时keys以及values也并在一起形成了矩阵KKK以及VVV。则attention的输出矩阵可以按照下述公式计算：

Attention(Q,K,V)=softmax(QKTdk−−√)VAttention(Q,K,V)=softmax(QKTdk)V

Attention(Q,K,V) = softmax({QK^T\\over {\\sqrt {d_k}}})V

![image](https://note.youdao.com/yws/api/personal/file/63021643358E47689B86ECA332EE2FBC?method=download&shareKey=952d59c0237db8aaa184381a877074fe)

### 2.4.2 Multi-Head Attention

本文结构中的Attention并不是简简单单将一个点乘的attention应用进去。作者发现先对queries，keys以及values进行hhh次不同的线性映射效果特别好。学习到的线性映射分别映射到dkdkd_k，dkdkd_k以及dvdvd_v维。分别对每一个映射之后的得到的queries，keys以及values进行attention函数的并行操作，生成dvdvdv维的output值。具体结构和公式如下。

MultiHead(Q,K,V)=Concat(head1,...,headh)MultiHead(Q,K,V)=Concat(head1,...,headh)

MultiHead(Q,K,V) = Concat(head\_1,...,head\_h)

where:headi=Attention(QWiQ,KWiK,VWiV)where:headi=Attention(QWiQ,KWiK,VWiV)

where: head\_i = Attention(Q{W\_i}^Q,K{W\_i}^K,V{W\_i}^V)

![image](https://note.youdao.com/yws/api/personal/file/30F703343EAA4A2EBCDBDAD352C39D21?method=download&shareKey=cc9219b0e1501110ad18e9779f27ec31)

### 2.4.3 模型中的attention

Transformer以三种不同的方式使用了多头attention。

1.  在encoder-decoder的attention层，queries来自于之前的decoder层，而keys和values都来自于encoder的输出。这个类似于很多已经提出的seq2seq模型所使用的attention机制。
2.  在encoder含有self-attention层。在一个self-attention层中，所有的keys,values以及queries都来自于同一个地方，本例中即encoder之前一层的的输出。
3.  类似的，decoder中的self-attention层也是一样。不同的是在scaled点乘attention操作中加了一个mask的操作，这个操作是保证softmax操作之后不会将非法的values连到attention中。

### 2.4.4 Feed Foreword

每层由两个支层，attention层就是其中一个，而attention之后的另一个支层就是一个前馈的网络。公式描述如下。  

FFN(x)=max(0,xW1+b1)W2+b2FFN(x)=max(0,xW1+b1)W2+b2

FFN(x) = max(0,xW\_1 + b\_1)W\_2 + b\_2


3\. 总结
======

模型的整体框架基本介绍完了，其最重要的创新应该就是Self-Attention和Multi-Head Attention的架构。在摒弃传统CNN和RNN的情况下，还能提高表现，降低训练时间。Transformer用于机器翻译任务，表现极好，可并行化，并且大大减少训练时间。并且也给我们开拓了一个思路，在处理问题时可以增加一种结构的选择。
