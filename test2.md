```python
import paddle
import paddle.fluid as fluid
import numpy
import sys
from __future__ import print_function

def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    predict = fluid.layers.fc(input=fc2, size=10, act='softmax')
    return predict

def inference_program():
    # The image is 32 * 32 with RGB representation.
    data_shape = [3, 32, 32]
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')

    #predict = resnet_cifar10(images, 32)
    predict = vgg_bn_drop(images) # un-comment to use vgg net
    return predict

def train_program():
    predict = inference_program()

    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, accuracy]

def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)

use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
trainer = fluid.Trainer(
    train_func=train_program,
    optimizer_func=optimizer_program,
    place=place)

# Each batch will yield 128 images
BATCH_SIZE = 128

# Reader for training
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.cifar.train10(), buf_size=50000),
    batch_size=BATCH_SIZE)

# Reader for testing. A separated data set for testing.
test_reader = paddle.batch(
    paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)


params_dirname = "image_classification_resnet.inference.model"

from paddle.v2.plot import Ploter

train_title = "Train cost"
test_title = "Test cost"
cost_ploter = Ploter(train_title, test_title)

step = 0
def event_handler_plot(event):
    global step
    if isinstance(event, fluid.EndStepEvent):
        cost_ploter.append(train_title, step, event.metrics[0])
        cost_ploter.plot()
        step += 1
    if isinstance(event, fluid.EndEpochEvent):
        avg_cost, accuracy = trainer.test(
            reader=test_reader,
            feed_order=['pixel', 'label'])
        cost_ploter.append(test_title, step, avg_cost)

        # save parameters
        if params_dirname is not None:
            trainer.save_params(params_dirname)

trainer.train(
    reader=train_reader,
    num_epochs=2,
    event_handler=event_handler_plot,
    feed_order=['pixel', 'label'])

# Prepare testing data.
from PIL import Image
import numpy as np
import os

def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)

    im = np.array(im).astype(np.float32)
    # The storage order of the loaded image is W(width),
    # H(height), C(channel). PaddlePaddle requires
    # the CHW order, so transpose them.
    im = im.transpose((2, 0, 1))  # CHW
    im = im / 255.0 #-1 - 1

    # Add one dimension to mimic the list format.
    im = numpy.expand_dims(im, axis=0)
    return im

cur_dir = os.getcwd()
img = load_image(cur_dir + '/03.image_classification/image/dog.png')
#img = load_image( './image/dog.png')

inferencer = fluid.Inferencer(infer_func=inference_program, param_path=params_dirname, place=place)

label_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# inference
results = inferencer.infer({'pixel': img})
#print(results[0])
print("infer results: %s" % label_list[np.argmax(results[0])+1])
```
