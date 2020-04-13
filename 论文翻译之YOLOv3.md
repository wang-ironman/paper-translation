# 摘要

我们向YOLO提供一些更新！我们做了一些小的设计改动，使之更好。我们还训练了一个相当棒的新网络。比上次大一点，但更准确。不过还是很快，别担心。320 × 320时，YOLOv3在28.2 mAP的情况下，以22ms的速度运行，与SSD一样精确，但速度快3倍。在旧的0.5 IOU mAP检测指标上，YOLOv3的表现是相当不错的。在Titan X上，它在51ms内达到57.9AP50，而RetinaNet在198ms内达到57.5AP50，性能相似，但速度快3.8倍。一如既往，所有代码都在https://pjreddie.com/yolo/上在线。

# 1. 引言

有时候你只需要打一年的电话就可以了，你知道吗？今年我没有做很多研究。花了很多时间在推特上。和GANs玩了一会儿。我有一点去年遗留下来的动力[12] [1]；我设法对YOLO做了一些改进。但是，老实说，没有什么比这个更有趣了，只是一堆小的改变，使它更好。我也帮助了别人的研究。

事实上，这就是我们今天来到这里的原因。我们有一个相机准备的最后期限[4]，我们需要对YOLO做一些随机更新，但我们没有来源。所以准备一份技术报告吧！

科技报告最棒的地方在于，他们不需要介绍，你们都知道我们来这里的原因。所以这篇导言的结尾将是论文其余部分的路标。首先我们会告诉你YOLOv3的处理是什么。那我们就告诉你怎么做。我们也会告诉你一些我们尝试过但没有成功的事情。最后，我们将思考这一切意味着什么。

# 2. The Deal

这就是YOLOv3的处理：我们主要从其他人那里得到好主意。我们还训练了一个新的分类器网络，比其他的分类器网络更好。我们将带你从头开始了解整个系统，这样你就可以完全理解了。

## 2.1 边界框预测

根据YOLO9000，我们的系统使用维度集群作为定位框来预测边界框[15]。网络为每个边界框预测4个坐标tx，ty，tw，th。如果单元格与图像左上角的偏移量为（cx，cy），并且前一个边框具有宽度和高度pw，ph，则预测对应于：

![image-20200413183539209](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413183539209.png)

在训练中，我们使用误差平方和损失。如果某个坐标预测的真实值为 ![image-20200413183708465](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413183708465.png)，我们的梯度是真实值（从真实框计算）减去我们的预测值：![image-20200413185659915](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413185659915.png)。这个真实值可以通过反转上面的方程来轻松计算。

YOLOv3使用logistic回归预测每个边界框的物体性评分。如果之前的边界框比之前的任何其他边界框与真实框重叠更多，则该值应为1。如果之前的边界框它不是最好的，但它确实与一个真实框重叠了超过某个阈值，我们忽略了这个预测，如[17]，我们使用的阈值是0.5。与[17]不同的是，我们的系统只为每个真实值指定一个边界框。如果一个边界框没有被指定给一个真实框，那么它不会对坐标或类预测造成任何损失，只有物体性。

## 2.2 分类预测

每个框使用多标签分类预测边界框可能包含的类。我们不使用softmax，因为我们发现它对良好的性能是不必要的，而只是使用独立的逻辑分类器。在训练过程中，我们使用二元交叉熵损失进行类预测。

当我们转移到更复杂的领域，比如Open Images Dataset[7]时，这个方式会有帮助。在这个数据集中有许多重叠的标签（即女人和人）。使用softmax可以假设每个框只有一个类，而这通常不是这样的。多标签方法可以更好地对数据建模。

## 2.3 跨尺度预测

YOLOv3预测3种不同尺度的box。我们的系统使用与金字塔网络相似的概念，从这些尺度中提取特征[8]。从我们的基本特征提取，我们添加了几个卷积层。最后一个预测三维tensor编码的边界框、物体性和类预测。在我们用COCO[10]做的实验中，我们在每个尺度上预测了3个box，因此对于4个边界box偏移量、1个物体性预测和80个类预测，张量是N × N ×[3 ×（4+1+80）]。

接下来，我们从前面的两层中提取特征图，并将其放大2倍。我们还从网络的早期获取一个特征映射，并使用cocat将其与我们的上采样特征合并。此方法允许我们从上采样的特征中获取更有意义的语义信息，并从早期的特征映射中获取更细粒度的信息。然后，我们再添加一些卷积层来处理这个组合的特征映射，并最终预测一个相似的张量，尽管现在的张量是原来的两倍。

我们再次执行相同的设计，以预测最终规模的框。因此，我们对第三个尺度的预测得益于所有先前的计算以及网络早期的细粒度特性。

我们仍然使用k-means聚类来确定我们的先验框。我们只是任意选择了9个簇和3个尺度，然后在尺度上均匀地划分簇。在COCO数据集上，9个聚类为：（10 × 13）；（16 × 30）；（33 × 23）；（30 × 61）；（62 × 45）；（59 × 119）；（116 × 90）；（156 × 198）；（373 × 326）。

## 2.4 特征提取

我们使用一个新的网络来进行特征提取。我们的新网络是YOLOv2，Darknet-19中使用的网络和新的残差网络之间的混合。我们的网络使用连续的3 × 3和1 × 1卷积层，但现在也有一些快捷的连接，并且明显更大。它有53个卷积层，所以我们称之为Darknet-53！

![image-20200413192344421](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413192344421.png)

这个新网络比Darknet19强大得多，但仍然比ResNet-101或ResNet-152更有效。
以下是一些ImageNet结果： 

![image-20200413192428699](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413192428699.png)

每个网络都接受相同设置的训练，并以256 × 256的single-crop精度进行测试。运行时间是在Titan X上以256 × 256测量的。因此，Darknet-53的性能与最先进的分类器不相上下，但浮点运算更少，速度更快。Darknet-53比ResNet-101好，速度快1.5倍。Darknet-53的性能与ResNet-152相似，速度快2倍。

Darknet-53还实现了每秒最高的浮点运算量。这意味着网络结构更好地利用了GPU，使其评估效率更高，因而速度更快。这主要是因为resnet有太多的层，效率不高。

## 2.5 训练

我们仍然在训练完整的图像，没有难例挖掘或任何东西。我们使用多尺度的训练，大量的数据增强，batch normalization，所有标准的东西。我们使用Darknet神经网络框架进行训练和测试[14]。

# 3. 我们是怎么做的

YOLOv3相当不错！见表3。

![image-20200413220412959](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413220412959.png)

就COCOs怪异mAP指标而言，它与SSD不相上下，但速度快3倍。在这个尺度上它还是有点落后于其他的像RetinaNet的模型。

然而，当我们在IOU=0.5（或图表中的AP50）处观察mAP的“旧”检测度量时，YOLOv3是非常强的。它几乎与RetinaNet相当，远高于SSD变体。这表明YOLOv3是一个非常强大的探测器，它擅长为物体制造像样的盒子。然而，随着IOU阈值的增加，性能显著下降，这表明YOLOv3难以使box与物体完全对齐。

在过去，YOLO与小物体上表现挣扎。然而，现在我们看到了这种趋势的逆转。通过新的多尺度预测，我们看到YOLOv3具有相对较高的APs性能。然而，它在中、大尺寸物体上的性能相对较差。需要更多的调查才能弄清真相。

当我们在AP50指标（见图5）上绘制精度与速度的关系图时，我们发现YOLOv3比其他检测系统具有显著的优势。也就是说，它更快更好。

![image-20200413223356548](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413223356548.png)

# 4. 我们尝试过的但是不起作用的

我们在做YOLOv3的时候试了很多东西。很多都没用。这是我们能记住的东西。

**Anchor box 的x，y偏移预测**。我们尝试使用普通的anchor box预测机制，其中使用线性激活将x，y偏移预测为框宽度或高度的倍数。我们发现这个公式降低了模型的稳定性，并且效果不太好。

**线性x；y预测，而不是logistic预测**。我们尝试使用线性激活来直接预测x，y偏移，而不是logistic激活。这导致了mAP的几个点下降。

**Focal loss**。我们试着用focal loss。我们的mAP掉了2个点。YOLOv3可能已经对focal loss试图解决的问题很健壮了，因为它有单独的物体性预测和条件类预测。因此，对于大多数例子来说，类预测没有损失？还是什么？我们不太确定。

**双IOU阈值和truth分配**。Faster RCNN在训练期间使用两个IOU阈值。如果一个预测与真实值重叠0.7，则它是一个正例；如果重叠大于0.3小于0.7，则它被忽略；对于所有真实框，重叠小于0.3，则它是一个负例。我们也尝试过类似的策略，但没有取得好的效果。

我们很喜欢我们目前的公式，它似乎至少在一个局部最优。有可能这些技巧最终会产生好的效果，也许他们只是需要一些调整来稳定训练。

# 5. 这些意味着什么

YOLOv3是一个很好的检测器。很快，很准确。在COCOmAP介于0.5和0.95 IOU之间的情况下，它并没有那么好。但在旧的0.5 IOU的检测标准上很好。

我们为什么要改变指标？COCO最初的论文只有一句话：“一旦评估服务器完成，将添加对评估指标的全面讨论”。Russakovsky等人报告说，人类很难区分0.3和0.5的IOU！“训练人类对IOU为0.3的边界框进行视觉检查，并将其与IOU为0.5的边界框区分开来，是非常困难的。”[18]如果人类很难分辨出两者之间的区别，那么这有多重要？

但也许一个更好的问题是：“既然我们有了这些检测器，我们该怎么处理呢？“做这项研究的很多人都在谷歌和Facebook。
我想至少我们知道这项技术掌握得很好，绝对不会被用来获取你的个人信息并卖给。。。。等等，你是说这正是它的用途？？哦。

好吧，其他大力资助计算机视觉研究的人是军方，他们从来没有做过像用新技术杀死很多人这样可怕的事情哦等等……1我非常希望大多数使用计算机视觉的人只是在用它做快乐的、好的事情，比如数国家公园里的斑马数量[13]，或者跟踪他们的猫在家里游荡。但是，计算机视觉已经开始被怀疑使用，作为研究人员，我们有责任至少考虑到我们的工作可能造成的危害，并想办法减轻它。我们欠世界那么多。

![image-20200413223318694](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413223318694.png)

![image-20200413223327767](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413223327767.png)

# 参考文献

[1] Analogy. Wikipedia, Mar 2018. 1
[2] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and
A. Zisserman. The pascal visual object classes (voc) challenge.
International journal of computer vision, 88(2):303–
338, 2010. 6
[3] C.-Y. Fu, W. Liu, A. Ranga, A. Tyagi, and A. C. Berg.
Dssd: Deconvolutional single shot detector. arXiv preprint
arXiv:1701.06659, 2017. 3
[4] D. Gordon, A. Kembhavi, M. Rastegari, J. Redmon, D. Fox,
and A. Farhadi. Iqa: Visual question answering in interactive
environments. arXiv preprint arXiv:1712.03316, 2017. 1
[5] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning
for image recognition. In Proceedings of the IEEE conference
on computer vision and pattern recognition, pages
770–778, 2016. 3
[6] J. Huang, V. Rathod, C. Sun, M. Zhu, A. Korattikara,
A. Fathi, I. Fischer, Z.Wojna, Y. Song, S. Guadarrama, et al.
Speed/accuracy trade-offs for modern convolutional object
detectors. 3
[7] I. Krasin, T. Duerig, N. Alldrin, V. Ferrari, S. Abu-El-Haija,
A. Kuznetsova, H. Rom, J. Uijlings, S. Popov, A. Veit,
S. Belongie, V. Gomes, A. Gupta, C. Sun, G. Chechik,
D. Cai, Z. Feng, D. Narayanan, and K. Murphy. Openimages:
A public dataset for large-scale multi-label and
multi-class image classification. Dataset available from
https://github.com/openimages, 2017. 2
[8] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and
S. Belongie. Feature pyramid networks for object detection.
In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, pages 2117–2125, 2017. 2, 3
[9] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Doll´ar.
Focal loss for dense object detection. arXiv preprint
arXiv:1708.02002, 2017. 1, 3, 4
[10] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan,
P. Doll´ar, and C. L. Zitnick. Microsoft coco: Common
objects in context. In European conference on computer
vision, pages 740–755. Springer, 2014. 2
[11] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-
Y. Fu, and A. C. Berg. Ssd: Single shot multibox detector.
In European conference on computer vision, pages 21–37.
Springer, 2016. 3
[12] I. Newton. Philosophiae naturalis principia mathematica.
William Dawson & Sons Ltd., London, 1687. 1
[13] J. Parham, J. Crall, C. Stewart, T. Berger-Wolf, and
D. Rubenstein. Animal population censusing at scale with
citizen science and photographic identification. 2017. 4
[14] J. Redmon. Darknet: Open source neural networks in c.
http://pjreddie.com/darknet/, 2013–2016. 3
[15] J. Redmon and A. Farhadi. Yolo9000: Better, faster, stronger.
In Computer Vision and Pattern Recognition (CVPR), 2017
IEEE Conference on, pages 6517–6525. IEEE, 2017. 1, 2, 3
[16] J. Redmon and A. Farhadi. Yolov3: An incremental improvement.
arXiv, 2018. 4
[17] S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: Towards
real-time object detection with region proposal networks.
arXiv preprint arXiv:1506.01497, 2015. 2
[18] O. Russakovsky, L.-J. Li, and L. Fei-Fei. Best of both
worlds: human-machine collaboration for object annotation.
In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, pages 2121–2131, 2015. 4
[19] M. Scott. Smart camera gimbal bot scanlime:027, Dec 2017.
4
[20] A. Shrivastava, R. Sukthankar, J. Malik, and A. Gupta. Beyond
skip connections: Top-down modulation for object detection.
arXiv preprint arXiv:1612.06851, 2016. 3
[21] C. Szegedy, S. Ioffe, V. Vanhoucke, and A. A. Alemi.
Inception-v4, inception-resnet and the impact of residual
connections on learning. 2017. 3