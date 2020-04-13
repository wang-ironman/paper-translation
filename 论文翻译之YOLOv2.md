# 摘要

我们介绍了一个最先进的实时目标检测系统YOLO9000，它可以检测超过9000个目标类别。首先，我们对YOLO检测方法提出了各种改进，既新颖又借鉴了前人的工作。改进后的YOLOv2是在如PASCAL VOC和COCO的标准检测任务上最先进的模型。YOLOv2使用一种新颖的多尺度训练方法，同一个YOLOv2模型可以在不同的size下运行，在速度和精度之间提供了一个简单的折衷。在67帧/秒的速度下，YOLOv2在VOC 2007上获得76.8的mAP。在40fps时，YOLOv2获得了78.6mAP，在运行速度显著提高的同时，它的性能超过了最新的方法，比如使用ResNet的SSD和Faster RCNN。最后提出了一种目标检测与分类联合训练的方法。利用该方法，我们在COCO检测数据集和ImageNet分类数据集上同时训练YOLO9000。我们的联合训练允许YOLO9000预测没有标记检测数据的物体类的检测。我们在ImageNet检测任务中验证了我们的方法。YOLO9000在ImageNet检测验证集上获得19.7mAP，尽管在200个类中只有44个有检测数据。对于不在COCO中的156个类，YOLO9000得到16.0mAP。但YOLO可以检测200多个类；它预测检测9000多个不同的物体类别。它仍然实时运行。

# 1. 引言

通用目标检测应该快速、准确，并且能够识别各种各样的目标。自从神经网络的引入，检测框架变得越来越快速和准确。然而，大多数检测方法仍然局限于一小部分目标。

与其他任务（如分类和标记）的数据集相比，当前的目标检测数据集受到限制。最常见的检测数据集包含数千到几十万个图像和几十到几百个标签[3] [10] [2]。分类数据集有数以百万计的图像，有几万或几十万个类别[20] [2]。

我们希望将检测扩展到物体分类的级别。然而，为检测图像贴标签比为分类或标签贴标签要昂贵得多（标签通常是用户免费提供的）。因此我们不太可能在不久的将来看到与分类数据集具有相同规模的检测数据集。

我们提出了一种新的方法来利用我们已经拥有的大量分类数据，并将其用于扩展现有检测系统的检测范围。我们的方法使用物体分类的层次视图，允许我们将不同的数据集组合在一起。

我们还提出了一种联合训练算法，允许我们在检测和分类数据上训练目标检测器。我们的方法利用标记的检测图像来学习精确定位目标，同时使用分类图像来增加词汇量和鲁棒性。

利用该方法，我们训练了YOLO9000，一个能够检测9000多种不同目标类别的实时目标检测器。首先，我们改进了基本的YOLO检测系统，产生了最先进的实时检测仪YOLOv2。然后利用我们的数据集组合方法和联合训练算法，在ImageNet的9000多个类和COCO的检测数据上训练一个模型。

![image-20200413154806758](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413154806758.png)

我们所有的代码和预训练模型都可以在http://pjreddie.com/yolo9000/上在线获得。

# 2. Better

与最先进的检测系统相比，YOLO有很多缺点。与Fast R-CNN相比，YOLO的定位误差分析表明YOLO的定位误差较大。此外，与基于区域建议的方法相比，YOLO的召回率相对较低。因此，我们主要关注在保持分类准确度的同时提高召回率和定位率。

计算机视觉通常趋向于更大、更深的网络[6] [18] [17]。更好的性能通常取决于训练更大的网络或将多个模型组合在一起。
然而，对于YOLOv2，我们需要一个更精确的检测器，但仍然是快速的。与其扩大我们的网络，不如简化网络，然后使表示更易于学习。我们将过去工作中的各种想法与自己新颖的概念结合起来，以提高YOLO的表现。
结果摘要见表2。

![image-20200413154906346](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413154906346.png)

**Batch Normallzation**。Batch Normalization导致了收敛性的显著改进，同时消除了对其他形式正则化的需要[7]。通过在YOLO的所有卷积层上添加Batch Normalization，mAP得到了2%以上的改进。Batch Normalization也有助于使模型正规化。通过Batch Normalization，我们可以在不过度拟合的情况下从模型中删除dropout。

**高分辨率分类器**。所有最先进的检测方法都使用在ImageNet上预先训练的分类器[16]。从AlexNet开始，大多数分类器对小于256 × 256[8]的输入图像进行操作。原始的YOLO在224 × 224处训练分类器网络，并将分辨率提高到448用于检测。这意味着网络必须切换到同时学习目标检测和调整新的输入分辨率。

对于YOLOv2，我们首先在ImageNet上的10个epochs内以448 × 448的分辨率对分类网络进行微调。这使得网络有机会调整其过滤器，以便更好地处理更高分辨率的输入。然后，我们在检测时对结果网络进行微调。这种高分辨率的分类网络使我们的mAP增加了近4%。

**Convolutional With Anchor Boxes**。YOLO在卷积特征提取器后使用全连接层直接预测边界框的坐标。Fast R-CNN使用手工挑选的priors[15]预测边界框，而不是直接预测坐标。Fast R-CNN中的区域建议网络（RPN）仅使用卷积层预测anchor box的偏移量和置信度。由于预测层是卷积的，RPN在特征图中的每个位置预测这些偏移。预测偏移而不是坐标简化了问题，使网络更容易学习。

我们从YOLO中移除全连接层，并使用anchor box来预测边界框。首先，我们去除一个池化层，使网络卷积层的输出具有更高的分辨率。我们还缩小了网络以操作416的输入图像，而不是448 × 448。我们这样做是因为我们希望在我们的特征图中有奇数个位置，这样就只有一个中心单元。物体，特别是大的物体，往往占据图像的中心，所以最好在中心有一个位置来预测这些物体，而不是四个在中心点附近的位置。YOLO卷积层将图像缩小32倍，因此通过使用416的输入图像，我们得到13 × 13的输出特征图。

当我们移动到anchor box时，我们还将类预测机制与空间位置分离，而是为每个anchor box预测类和物体性。跟随YOLO，物体性预测仍然预测真实值和预测框的IOU，类预测值预测在给定那是一个物体情况下该类的条件概率。

使用anchor box，我们的精度会有小的下降。YOLO在每张图片只预测98个box，但是我们的使用anchor box的模型预测有一千多个box。没有anchor box，我们的中间模型可以得到69.5的mAP，召回率为81%。使用anchor，我们的模型可以得到69.2的mAP，召回率为88%。尽管mAP减少了，但召回率的增加意味着我们的模型还有更大的改进空间。

**Dimension Clusters**。我们在与YOLO一起使用anchor box时遇到了两个问题。首先是box的尺寸是手工挑选的。网络可以学会适当地调整box，但是如果我们从网络开始选择更好的先验框，我们可以使网络更容易学会预测好的检测。

我们在训练集的边界框上运行k-means聚类，而不是手动选择先验框，从而自动地找到好的先验框。如果我们使用欧几里德距离的标准k-means，则较大的box比较小的box产生更多的误差。然而，我们真正想要的是能够获得高分的先验框，这与box的大小无关。因此，对于我们的距离度量，我们使用：

![image-20200413164526890](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413164526890.png)

我们对k的各种值运行k-means，并绘制具有最近质心的平均IOU，见图2。

![image-20200413164606243](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413164606243.png)

我们选择k=5作为模型复杂度和高召回率之间的一个很好的折衷。簇质心明显不同于人工选取的anchor box。少了一些又短又宽的盒子，多了一些又高又窄的盒子。

我们比较了平均IOU与我们的聚类策略的最近先验框，以及表1中的手动选择anchor box。
在5个先验框条件下，质心的表现与9个anchor box相似，平均IOU一个为61.0，另一个为60.9。如果我们使用9个质心，我们会看到更高的平均IOU。这表明，使用k-means生成我们的边界框可以以更好的表示方式启动模型，并使任务更易于学习。

![image-20200413155515122](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413155515122.png)

**直接定位预测**。当使用anchor box和YOLO时，我们遇到了第二个问题：模型不稳定性，特别是在早期迭代期间。大多数不稳定性来自于预测box的（x，y）位置。在区域建议网络中，网络预测值tx和ty以及（x，y）中心坐标的计算公式为：

![image-20200413165240774](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413165240774.png)

例如，预测tx=1将使anchor box的宽度向右移动，预测tx=-1将使anchor box的宽度向左移动相同的量。

此公式不受约束，因此任何anchor box都可以在图像中的任何点结束，而不管预测框的位置如何。在随机初始化的情况下，模型需要很长时间才能稳定下来，以预测合理的偏移量。

我们不是按照YOLO的方法来预测偏移，而是预测相对网格单元位置的位置坐标。这限制了真实值在0和1之间。我们使用logistic激活来限制网络的预测在这个范围内。

该网络在输出特征图的每个网格单元预测5个边界框。网络为每个边界框预测5个坐标tx、ty、tw、th和to。如果单元格与图像左上角的偏移量为（cx，cy），并且prior 边界框具有宽度和高度pw，ph，则预测对应于：

![image-20200413170118015](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413170118015.png)

因为我们限制了位置预测，所以参数更容易学习，使网络更加稳定。使用维度聚类和直接预测边界框中心位置比使用anchor box的版本提高了约5%。

![image-20200413172049690](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413172049690.png)

**细粒度特征**。这个改进的YOLO预测在13 × 13特征图上的检测。虽然这对于大型物体来说已经足够了，但它可能受益于用于定位较小物体的细粒度特性。Faster R-CNN和SSD都在网络中的各种特征图上运行他们的提议网络，以获得一系列的分辨率。我们采用了不同的方法，只需添加一个passthrough层，它以26 × 26的分辨率带来了先前层的特征。

与ResNet中的身份映射（identity mappings）类似，passthrough层通过将相邻特征叠加到不同的channels而不是空间位置，将高分辨率特征与低分辨率特征连接起来。这将26 × 26 × 512的特征图转换为13 × 13 × 2048的特征图，该特征图可以与原先的特征连接。我们的检测器运行在这个扩展的特征图之上，因此它可以访问细粒度特征。这使性能略微提高了1%。

**多尺度训练**。原始的YOLO使用448 × 448的输入分辨率。随着anchor box的增加，我们把分辨率改为416 × 416。但是，由于我们的模型只使用卷积层和池化层，因此可以动态调整大小。我们希望YOLOv2能够在不同大小的图像上运行，因此我们将其训练为模型。

我们不是固定输入图像的大小，而是每隔几次迭代就改变一次网络。每10 batch我们的网络随机选择一个新的图像尺寸大小。由于我们的模型下采样率为32，我们从以下32的倍数中提取：（320，352，...，608）。因此最小的选项是320 × 320，最大的是608 × 608。我们将网络调整到这个维度并继续训练。

这种机制迫使网络学会在各种输入维度上进行好的预测。这意味着同一个网络可以在不同分辨率下预测检测结果。网络在较小的规模下运行更快，因此YOLOv2在速度和精度之间提供了一个简单的折衷。

在低分辨率下，YOLOv2是一种廉价（计算）、相当精确的探测器。在288 × 288上，它的运行速度超过90 FPS且有着几乎与Fast R-CNN一样的mAP。这使得它非常适合于较小的GPU、高帧率视频或多个视频流。

在高分辨率下，YOLOv2是一种最先进的探测器，在VOC 2007上有78.6mAP，同时仍以高于实时速度运行。YOLOv2的在VOC 2007上与其他框架的比较见表3，图4：

![image-20200413171605083](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413171605083.png)

![image-20200413171646181](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413171646181.png)

**进一步的实验**。我们训练YOLOv2在VOC 2012上进行检测。表4显示了YOLOv2与其他最新检测系统的比较性能。
YOLOv2在运行速度远快于竞争方法的同时实现了73.4mAP。我们还在COCO上进行了训练，并与表5中的其他方法进行了比较。在VOC度量（IOU=0.5）上，YOLOv2得到44.0的mAP，与SSD和 Faster R-CNN相当。

![image-20200413171904271](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413171904271.png)

![image-20200413171923857](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413171923857.png)

# 3. Faster

我们希望检测准确，但我们也希望它快速。大多数检测应用，如机器人或自动驾驶汽车，都依赖于低延迟预测。为了最大限度地提高性能我们将YOLOv2设计得从一开始就很快。

大多数检测框架依赖于VGG-16作为基本特征抽取器[17]。VGG-16是一个强大、准确的分类网络，但它有不必要的复杂。VGG-16的卷积层需要对224 × 224分辨率下的单个图像进行306.9亿次浮点运算。

YOLO框架使用基于Googlenet架构的定制网络[19]。这个网络比VGG-16要快，仅用85.2亿次操作进行一次前向传播。然而，它的精度比VGG16稍差。对于single-crop，在ImageNet上，224 × 224的top-5精度，YOLO的定制模型88.0%，而VGG-16则为90.0%。

**Darknet-19**。我们提出了一个新的分类模型作为YOLOv2的基础。我们的模型建立在网络设计的前期工作和该领域的常识的基础上。与VGG模型类似，我们通常使用3 × 3的过滤器，在每个池化步骤之后通道数量增加一倍[17]。在网络中的网络（NIN）工作之后，我们使用全局平均池化进行预测，并使用1 × 1过滤器压缩3 × 3卷积之间的特征表示[9]。我们使用batch normalization来稳定训练，加速收敛，并使模型正则化[7]。

我们的最终模型，称为Darknet-19，有19个卷积层和5个maxpooling层。有关详细说明，请参见表6.

![image-20200413173257091](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413173257091.png)

Darknet-19只需要55.8亿次操作就可以处理图像，但在ImageNet上达到72.9%的top-1精度和91.2%的top-5精度。

**分类训练**。我们使用Darknet神经网络框架，在标准ImageNet 1000类分类数据集上训练网络，使用随机梯度下降（起始学习率为0.1）、多项式速率衰减（幂为4）、权重衰减（权重衰减为0.0005）和动量（动量为0.9）对160个epoch进行分类[13]。在训练过程中，我们使用标准的数据增强技巧，包括随机裁剪、旋转、色调、饱和度和曝光偏移。

如上所述，在224 × 224的初始图像训练之后，我们对网络进行了更大size（448）的微调。对于这种微调，我们使用上述参数进行训练，但只训练了10个阶段，并以10^-3的学习速率开始。在这个更高的分辨率下，我们的网络达到了76.5%的top-1精度和93.3%的top-5精度。

**检测训练**。我们通过删除最后一个卷积层来修改该网络以进行检测，我们不是添加三个3 × 3卷积层（每个卷积层具有1024个滤波器），而是最后添加一个1 × 1卷积层以及我们需要检测的输出数量（滤波器的数量）。对于VOC，我们预测5个框，每个框有5个坐标，每个框有20个类，因此125个过滤器。我们还添加了一个从最后的3 × 3 × 512层到第二个到最后的卷积层的passthrough层，以便我们的模型可以使用细粒度特征。

我们对网络进行160个epochs的训练，初始学习率为10^-3，在60和90个epoch将学习率除以10。我们使用0.0005的重量衰减和0.9的动量。我们使用了类似于YOLO和SSD的随机裁剪、颜色变化等数据增强方法，对COCO和VOC使用了相同的训练策略。

# 4. Stronger

提出了一种分类检测数据联合训练机制。我们的方法使用标记检测的图像来学习特定的检测信息，如边界框坐标预测和物体性，以及如何对常见物体进行分类。它使用只有类标签的图像来扩展它可以检测到的类别数。

在训练过程中，我们混合了来自检测和分类数据集的图像。当我们的网络看到标记为检测的图像时，我们可以基于完全的YOLOv2损失函数进行反向传播。当它看到一个分类图像时，我们只会反向传播来自体系结构的分类的特定部分的损失。

这种方法带来了一些挑战。检测数据集只有通用物体和通用标签，如“狗”或“船”。分类数据集的标签范围更广、更深。ImageNet有超过一百种狗，包括“诺福克犬”、“约克郡犬”和“贝灵顿犬”。如果我们想在两个数据集上进行训练，我们需要一种连贯的方法来合并这些标签。

大多数分类方法在所有可能的分类中使用softmax层来计算最终的概率分布。使用softmax是假设类是互斥的。这就给组合数据集带来了问题，例如，您不想使用此模型组合ImageNet和COCO，因为类“Norfolk terrier”和“dog”不是互斥的。

相反，我们可以使用多标签模型来组合不假定互斥的数据集。这种方法忽略了我们所知道的关于数据的所有结构，例如，所有COCO类都是互斥的。

**分级分类**。ImageNet标签是从WordNet中提取出来的，WordNet是一个语言数据库，用于构建概念及其关联方式[12]。在wordnet中，“Norfolk terrier”和“Yorkshire terrier”都是“terrier”的下称，terrier是“猎犬”的一种，是“狗”的一种，是“犬”的一种，等等。大多数分类方法对标签采用扁平结构，但是对于组合数据集，结构正是我们所需要的。

WordNet的结构是有向图，而不是树，因为语言是复杂的。例如，“狗”既是“狗”的一种类型，又是“家畜”的一种类型，在wordnet中都是语法集。我们不使用完整的图结构，而是根据ImageNet中的概念构建一个层次树来简化问题。

为了构建这个树，我们检查ImageNet中的可视名词，并查看它们通过WordNet图到根节点的路径，在本例中是“实物”。许多语法集只有一条通过图的路径，因此首先我们将所有这些路径添加到我们的树中。然后我们反复检查我们留下的概念，并尽可能少地添加使树生长的路径。因此，如果一个概念有两条到根的路径，一条路径将向我们的树添加三条边，而另一条只添加一条边，那么我们选择较短的路径。

最终的结果是WordTree，一个视觉概念的层次模型。为了使用WordTree进行分类，我们预测每个节点上的条件概率，即给定该系统集（sysnsets）的每个子词的概率。例如，在“terrier”节点，我们预测：

![image-20200413174841536](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413174841536.png)

如果我们想计算一个特定节点的绝对概率，我们只需沿着路径从树到根节点，然后乘以条件概率。因此，如果我们想知道一张图片是不是诺福克猎犬，我们计算：

![image-20200413174910285](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413174910285.png)

为了分类，我们假设图像包含一个物体：Pr（实物）=1。

为了验证这种方法，我们在使用1000类ImageNet构建的WordTree上训练Darknet-19模型。为了构建WordTree1k，我们添加了所有中间节点，这些节点将标签空间从1000扩展到1369。在训练过程中，我们将真实值标记在树上，这样，如果一个图像被标记为“诺福克猎犬”，它也被标记为“狗”和“哺乳动物”等。为了计算条件概率，我们的模型预测了一个1369值的向量，并且我们计算了作为同一概念下位词的所有系统集的softmax，见图5。

![image-20200413175024878](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413175024878.png)

使用与之前相同的训练参数，我们的分层Darknet-19达到71.9%的top-1精度和90.4%的top-5精度。尽管增加了369个额外的概念，并且我们的网络预测了一个树结构，但我们的准确率只是略有下降。以这种方式执行分类也有一些好处。在新的或未知的物体类别上，性能会正常下降。例如，如果网络看到一只狗的图片，但不确定它是哪种类型的狗，它仍然会以高度的信心预测“狗”，但在下义词之间传播的信心较低。

这种方式也适用于检测。现在，我们不再假设每个图像都有一个物体，而是使用YOLOv2的物体预测器来给出Pr（实物）的值。探测器预测一个边界框还有概率树。我们向下遍历树，在每次拆分时采用最高的置信度路径，直到达到某个阈值，然后预测该物体类。

**数据集与WordTree的组合**。我们可以使用WordTree以合理的方式将多个数据集组合在一起。我们只需将数据集中的类别映射到树中的语法集。图6显示了一个使用WordTree合并ImageNet和COCO中的标签的示例。

![image-20200413175422260](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413175422260.png)

WordNet是非常多样化的，所以我们可以在大多数数据集中使用这种技术。

**联合分类和检测**。现在我们可以使用WordTree组合数据集，我们可以训练我们的分类和检测联合模型。我们希望训练一个超大规模的检测器，因此我们使用COCO检测数据集和来自完整ImageNet版本的前9000个类创建组合数据集。我们还需要对我们的方法进行评估，以便添加ImageNet detection challenge中尚未包含的任何类。这个数据集对应的WordTree有9418个类。ImageNet是一个大得多的数据集，因此我们通过对COCO进行过采样来平衡数据集，这样ImageNet只比COCO大4:1倍。

我们用这个数据集训练YOLO9000。我们使用基本的YOLOv2架构，但是只有3个先验框而不是5个先验框来限制输出大小。当我们的网络看到检测图像时，我们会像正常情况一样反向传播loss。对于分类损失，我们只在标签的相应级别或更高级别上反向传播损失。例如，如果标签是“狗”，我们会将任何错误分配给树下更远的预测，“德国牧羊犬”与“黄金猎犬”，因为我们没有这些信息。

当它看到分类图像时，我们只会反向传播分类损失。要做到这一点，我们只需找到预测该类的最高概率的边界框，然后计算其预测树上的损失。我们还假设预测框至少与真实值重叠0.3的iou，并基于此假设反向传播目标损失。通过这种联合训练，YOLO9000学习使用COCO中的检测数据在图像中查找物体，并学习使用ImageNet中的数据对各种物体进行分类。

我们评估YOLO9000在ImageNet检测任务中的表现。ImageNet的检测任务与COCO共享44个目标类别，这意味着YOLO9000只看到了大多数测试图像的分类数据，而没有检测数据。YOLO9000在不相交的156个物体类上得到了19.7mAP和16.0mAP，而这些物体类从未见过任何标记的检测数据。此映射比DPM获得的结果要高，但YOLO9000仅在部分监督的情况下在不同的数据集上进行训练[4]。它还同时检测9000个其他目标类别，全部是实时的。

当我们分析YOLO9000在ImageNet上的表现时，我们看到它很好地学习了新物种的动物，但却在学习诸如服装和设备等类别上很挣扎。

![image-20200413180112693](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200413180112693.png)

新的动物更容易学习，因为物体性预测很好地概括了COCO中的动物。相反，COCO没有任何类型的服装的边界框标签，只有人穿的，所以YOLO9000很难为“太阳镜”或“泳裤”等类别的服装模型化。

# 5. 总结

本文介绍了实时检测系统YOLOv2和YOLO9000。YOLOv2是最先进的，在各种检测数据集上比其他检测系统更快。此外，它可以运行在不同的图像大小，以提供速度和精度之间的平滑权衡。

YOLO9000是一个通过联合优化检测和分类来检测9000多个目标类别的实时框架。我们使用WordTree来组合来自不同来源的数据，并使用我们的联合优化技术在ImageNet和COCO上同时进行训练。YOLO9000是缩小检测和分类之间数据集大小差距的有力一步。

我们的许多技术都是在目标检测之外推广的。我们对ImageNet的WordTree表示为图像分类提供了更丰富、更详细的输出空间。使用分层分类的数据集组合在分类和分割领域将是有用的。像多尺度训练这样的训练技术可以在各种视觉任务中提供好处。

在以后的工作中，我们希望使用类似的技术来进行弱监督图像分割。我们还计划在训练期间使用更强大的匹配策略将弱标签分配给分类数据，以提高检测结果。计算机视觉拥有大量的标记数据。我们将继续寻找方法，将不同来源和结构的数据汇集在一起，以建立更强大的视觉世界模型。

# 参考文献

[1] S. Bell, C. L. Zitnick, K. Bala, and R. Girshick. Insideoutside net: Detecting objects in context with skip
pooling and recurrent neural networks. arXiv preprintarXiv:1512.04143, 2015. 6
[2] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database.
In Computer Vision and Pattern Recognition, 2009. CVPR 2009.IEEE Conference on, pages 248–255. IEEE, 2009. 1
[3] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman. The pascal visual object classes (voc) challenge.International journal of computer vision, 88(2):303–338, 2010. 1
[4] P. F. Felzenszwalb, R. B. Girshick, and D. McAllester.Discriminatively trained deformable part models, release 4.
http://people.cs.uchicago.edu/ pff/latent-release4/. 8
[5] R. B. Girshick. Fast R-CNN. CoRR, abs/1504.08083, 2015.4, 5, 6
[6] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385,2010.2, 4, 5
[7] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift.
arXiv preprint arXiv:1502.03167, 2015. 2, 5
[8] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In
Advances in neural information processing systems, pages 1097–1105, 2012. 2
[9] M. Lin, Q. Chen, and S. Yan. Network in network. arXiv preprint arXiv:1312.4400, 2013. 5
[10] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan,P. Doll´ar, and C. L. Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision, pages 740–755. Springer, 2014. 1, 6
[11] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, and S. E. Reed. SSD: single shot multibox detector. CoRR, abs/1512.02325,2011.4, 5, 6
[12] G. A. Miller, R. Beckwith, C. Fellbaum, D. Gross, and K. J. Miller. Introduction to wordnet: An on-line lexical database. International journal of lexicography, 3(4):235–244, 1990.6
[13] J. Redmon. Darknet: Open source neural networks in c.http://pjreddie.com/darknet/, 2013–2016. 5
[14] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You only look once: Unified, real-time object detection. arXiv
preprint arXiv:1506.02640, 2015. 4, 5
[15] S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: Towards real-time object detection with region proposal networks.arXiv preprint arXiv:1506.01497, 2015. 2, 3, 4, 5,6
[16] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein,
A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer
Vision (IJCV), 2015. 2
[17] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint
arXiv:1409.1556, 2014. 2, 5
[18] C. Szegedy, S. Ioffe, and V. Vanhoucke. Inception-v4, inception-resnet and the impact of residual connections on
learning. CoRR, abs/1602.07261, 2016. 2
[19] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed,D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich.
Going deeper with convolutions. CoRR, abs/1409.4842,2012.5
[20] B. Thomee, D. A. Shamma, G. Friedland, B. Elizalde, K. Ni,D. Poland, D. Borth, and L.-J. Li. Yfcc100m: The new
data in multimedia research. Communications of the ACM,59(2):64–73, 2016. 1