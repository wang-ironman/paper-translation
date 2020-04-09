# 摘要
目前最先进的物体检测网络依靠区域提议（proposal）算法来假设物体的位置。 SPPnet [1]和Fast R-CNN [2]之类的进步减少了这些检测网络的运行时间，但也揭示了区域提议计算存在瓶颈。在这项工作中，我们引入了一个区域提议网络（RPN），该区域提议网络与检测网络共享全图像卷积特征，从而实现几乎免费（计算）的区域提议。RPN是一个全卷积的网络，可以同时预测每个位置的目标边界和目标得分。对RPN进行了端到端的训练，以生成高质量的区域建议，Fast R-CNN将这些区域提议用于检测。通过共享RPN和Fast R-CNN的卷积特征，我们将RPN和Fast R-CNN进一步合并为一个网络-使用最近流行的神经网络术语：“注意力”机制，RPN组件告诉统一网络要看（注重）的地方。对于非常深的VGG-16模型[3]，我们的检测系统在GPU上具有5fps（包括所有步骤）的帧频，同时在PASCAL VOC 2007、2012和MS COCO数据集上达到了最优（SOTA）的目标检测精度。每个图像仅包含300个建议。在ILSVRC和COCO 2015竞赛中，Faster R-CNN和RPN是多个比赛中第一名获胜作品的基础。代码已公开提供。
索引词-目标检测，区域提议，卷积神经网络

# 1. 引言（Introdution）
区域提议方法（例如[4]）和基于区域的卷积神经网络（RCNN）[5]的成功推动了目标检测的最新进展。尽管基于区域的CNN在计算上很昂贵，如最初在[5]中发展的，但由于[1]，[2]在提议之间共享卷积，因此其计算成本已大大降低。最新的改进，Fast R-CNN [2]，在忽略区域建议花费的时间时，使用非常深的网络[3]实现了接近实时的速度。现在，建议是最先进的检测系统中的测试时间计算瓶颈。
区域提议方法通常依赖于低质量（inexpensive）的特征和经济的推理方案。选择性搜索[4]是最流行的方法之一，它根据工程化的底层特征贪婪地合并超像素。然而，与高效的检测网络相比[2]，选择性搜索的速度要慢一个数量级，运行在CPU中每张图像花费2秒。 EdgeBoxes [6]当前提供建议质量和速度之间的最佳权衡，每张图像0.2秒。尽管如此，区域提议步骤仍然消耗与检测网络一样多的运行时间。
可能会注意到，基于区域的快速CNN充分利用了GPU的优势，而研究中使用的区域提议方法则是在CPU上实现的，因此这种运行时比较是不公平的。加速提议计算的一种明显方法是为GPU重新实现。这可能是一种有效的工程解决方案，但是重新实现会忽略下游检测网络，因此会错过共享计算的重要机会。
 在本文中，我们证明了算法的变化（使用深度卷积神经网络计算建议）导致了一种优雅而有效的解决方案，考虑到检测网络的计算，建议计算几乎是免费的。为此，我们引入了与最先进的目标检测网络[1]，[2]共享卷积层的新颖的区域提议网络（RPN）。通过在测试时共享卷积，计算建议的边际成本很小（例如，每张图片10毫秒）。
我们的观察结果是，基于区域的检测器（如Fast RCNN）使用的卷积特征图也可用于生成区域建议。在这些卷积特征之上，我们通过添加一些其他卷积层来构建RPN，这些卷积层同时回归规则网格上每个位置的区域边界和物体性得分。因此，RPN是一种全卷积网络（FCN）[7]，可以专门针对生成检测建议的任务进行端到端训练。
RPN旨在以各种尺度（比例）和宽高比有效预测区域提议。与使用图像金字塔（图1，a）或过滤器金字塔（pyramids of filters）（图1，b）的流行方法[8]，[9]，[1]，[2]相比，我们介绍了新颖的“锚”盒作为多种比例和宽高比的参考。我们的方案可以看作是回归参考（regression references）的金字塔（图1，c），它避免了枚举（enumerating）具有多个比例或宽高比的图像或过滤器。当使用单尺度图像进行训练和测试时，该模型表现良好，从而提高了运行速度。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191230201531722.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
**图1：**解决图像多种尺度和大小的不同方案。 （a）建立图像和特征图的金字塔，并在所有比例下运行分类器。 （b）具有多个比例/大小的过滤器金字塔在特征图上运行。 （c）我们在回归函数中使用参考boxes的金字塔。

为了将RPN与Fast R-CNN [2]目标检测网络统一起来，我们提出了一种训练方案，该方案在对区域建议任务进行微调与对目标检测进行微调之间交替，同时保持建议不变。该方案快速收敛，并产生具有两个任务之间共享的卷积特征的统一网络。
我们在PASCAL VOC检测基准[11]上全面评估了我们的方法，其中使用RPN的Fast R-CNN产生的检测精度要优于使用选择性搜索的Fast R-CNN。同时，我们的方法在测试时几乎免除了“选择性搜索”的所有计算负担-proposal的有效运行时间仅为10毫秒。使用优质（expensive）的非常深的模型[3]，我们的检测方法在GPU上的帧速率仍然为5fps（包括所有步骤），因此这是一个在速度和准确性方面都很实用的目标检测系统。我们还报告了MS COCO数据集的结果[12]，并使用COCO数据研究了PASCAL VOC的改进。代码已在https://github.com/shaoqingren/faster_ rcnn（MATLAB代码）和https://github.com/rbgirshick/py-faster-rcnn（Python代码）中公开提供。
该手稿的初步版本先前已发布[10]。从那时起，RPN和Faster R-CNN的框架已被采用并推广到其他方法，例如3D目标检测[13]，基于零件的检测（part-based）[14]，实例分割[15]和图像字幕(captioning)[16]。我们的快速有效的目标检测系统也已建立在商业系统中，例如Pinterests [17]，据报道用户参与度有所提高。
在ILSVRC和COCO 2015竞赛中，Faster R-CNN和RPN是ImageNet检测，ImageNet定位（localization），COCO检测和COCO分割中几个第一名的基础[18]。RPN完全学会了根据数据提议区域，因此可以轻松地从更深，更具表现力的特征（例如[18]中采用的101层残差网络）中受益。在这些比赛中，其他一些领先的参赛者也使用了Faster R-CNN和RPN。这些结果表明，我们的方法不仅是一种实用的高性价比解决方案，而且还是提高目标检测精度的有效途径。
# 2. 相关工作
**目标建议（Object Proposals）**
关于目标建议方法的文献很多。可以在[19]，[20]，[21]中找到目标建议方法的综合研究和比较。广泛使用的目标建议方法包括基于超像素分组的方法（例如，选择性搜索[4]，CPMC [22]，MCG [23]）和基于滑动窗口的方法（例如，窗口中的物体[24]，EdgeBoxes [  6]）。目标建议方法用于作为独立于检测器的外部模块（例如，选择性搜索[4]目标检测器，RCNN [5]和Fast R-CNN [2]）。
 **用于目标检测的深度网络**
 R-CNN方法[5]端到端训练CNN将提议区域分类为物体类别或背景。R-CNN主要充当分类器，并且不预测目标边界（通过边界框回归进行精炼除外）。它的准确性取决于区域提议模块的性能（请参见[20]中的比较）。几篇论文提出了使用深度网络预测目标边界框的方法[25]，[9]，[26]，[27]。
在OverFeat方法[9]中，训练了一个全连接层来预测假设单个目标的定位任务的框坐标。然后将全连接层转换为卷积层，以检测多个类特定的物体。MultiBox方法[26]，[27]从网络中生成区域提议，该网络的最后一个全连接层同时预测多个与类无关的boxes，从而概括（generalizing）了OverFeat的“single-box”方式。这些与类无关的boxes用作R-CNN的建议[5]。与我们的全卷积方案相比，MultiBox建议网络适用于单个图片裁剪（crop）或多个大图片裁剪（crop）（例如224* 224）。MultiBox在提议和检测网络之间不共享特征。我们稍后将在我们的方法中更深入地讨论OverFeat和MultiBox。与我们的工作同时开发的DeepMask方法[28]，用于学习分割建议。
卷积的共享计算[9]，[1]，[29]，[7]，[2]已吸引了越来越多的关注，以进行有效而准确的视觉识别。 OverFeat论文[9]从图像金字塔计算卷积特征，以进行分类，定位和检测。共享卷积特征图上的自适应大小池化（SPP）[1]被开发用于有效的基于区域的目标检测[1]，[30]和语义分割[29]。Fast R-CNN [2]可以对共享卷积特征进行端到端检测器训练，并显示出令人信服的准确性和速度。


# 3. Faster R-CNN
 我们的目标检测系统称为Faster R-CNN，它由两个模块组成。第一个模块是提出区域的深层全卷积网络，第二个模块是使用提出的区域的Fast R-CNN检测器[2]。整个系统是用于目标检测的单个统一网络（图2）。RPN模块使用最近流行的神经网络术语：带有“注意力” [31]机制，RPN模块可以告诉Fast R-CNN模块在哪里查看（注重哪部分）。在第3.1节中，我们介绍了用于区域提议的网络的设计和属性。在3.2节中，我们开发了用于训练具有共享特征的两个模块的算法。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191231162214260.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
**图2：**Faster R-CNN是一个单个的统一的目标检测网络。RPN模块相当于这个统一网络的‘注意力’。


## 3.1 区域提议网络
区域提提议网络（RPN）接收（任意大小的）图像作为输入，并输出一组矩形的目标提议，每个提议均具有物体性得分（“Region” is a generic term and in this paper we only consider rectangular regions, as is common for many methods (e.g., [27], [4],[6]). “Objectness” measures membership to a set of object classes
vs. background.）。我们使用全卷积网络对该过程进行建模[7]，我们将在下文中对此进行描述这个部分。因为我们的最终目标是与Fast R-CNN目标检测网络共享计算[2]，所以我们假设两个网络共享一组共同的卷积层。在我们的实验中，我们研究了具有5个可共享卷积层的Zeiler和Fergus模型[32]（ZF）以及具有13个可共享卷积层的Simonyan和Zisserman模型[3]（VGG-16）。
为了生成区域建议，我们在最后共享的卷积层输出的卷积特征图上滑动一个小型网络。这个小网络将输入卷积特征图的n×n空间窗口作为输入。每个滑动窗口都映射到一个较低维的特征（ZF为256-d，VGG为512-d，后面是ReLU [33]）。此特征被馈入（fed into）两个同级（sibling）的全连接层---框回归层（reg）和框分类层（cls）。在本文中，我们使用n = 3，注意输入图像上的有效感受野很大（ZF和VGG分别为171和228像素）。在图3的单个位置（左）显示了此微型网络。请注意，由于微型网络以滑动窗口的方式运行，因此全连接层将在所有空间位置上共享。该体系结构自然是由n×n卷积层和两个同级1×1卷积层（分别用于reg和cls）实现的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191231164234325.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
**图3：**左：区域提议网络（RPN）。右：在PASCAL VOC 2007测试中使用RPN建议的检测示例。我们的方法可以检测各种比例和宽高比的物体。

### 3.1.1 Anchors
在每个滑动窗口位置，我们同时预测多个区域提议，其中每个位置的最大可能提议数目表示为k。因此，reg层有4k个输出，对k个框的坐标进行编码，而cls层则输出2k个分数，这些分数估计每个提议（For simplicity we implement the cls layer as a two-class softmax layer. Alternatively, one may use logistic regression to
produce k scores.）的物体或非物体的概率。相对于k个参考框（称为anchors），对k个建议进行了参数化。anchor位于相关滑动窗口的中心，并与比例和宽高比相关联（图3，左）。默认情况下，我们使用3个尺度和3个宽高比，在每个滑动位置产生k = 9个anchors。对于大小为W×H（通常为2,400）的卷积特征图，总共有W×H×k个anchors。
**平移不变的anchors**
我们方法的一个重要特性是，就anchors和计算相对于anchors的提议的函数而言，它是平移不变的。如果平移（translate）了图像中的一个目标，则该提议也应进行平移，并且相同的特征应能够在任一位置预测该提议。此平移不变属性由我们的方法（As is the case of FCNs [7], our network is translation invariant up to the network’s total stride.）保证。作为比较，MultiBox方法[27]使用k均值生成800个anchors，这些anchors不是平移不变的。因此，MultiBox不保证平移目标时会生成相同的建议。
平移不变属性还减小了模型大小。MultiBox具有（4 +1）× 800维 的全连接输出层，而在k = 9个anchors的情况下，我们的方法具有一个（4 + 2）×9维 的卷积输出层。最终，我们的输出层具有2.8 ×10的4次方个参数（VGG-16为512×（4 + 2）×9个），比具有6.1 ×10的6次方个参数（MultiBox里的GoogleNet为1536×（4 + 1）×800个）的MultiBox输出层少了两个数量级。如果考虑特征投影层，我们的建议层的参数仍然比MultiBox（Considering the feature projection layers, our proposal layers’ parameter count is 3 × 3 × 512 × 512 + 512 × 6 × 9 = 2.4 × 10^6;MultiBox’s proposal layers’ parameter count is 7 × 7 × (64 + 96 +64 + 64) × 1536 + 1536 × 5 × 800 = 27 × 10的6次方）少一个数量级。我们希望我们的方法在较小的数据集（如PASCAL VOC）上过拟合的风险较小。
**多尺度anchors作为回归参考**
我们的anchor设计提出了解决多种尺度（和宽高比）的新颖方案。如图1所示，有两种流行的多尺度预测方法。第一种方法是例如在DPM中 [8]使用的基于图像/特征金字塔和[9]，[1]，[2]中基于CNN的方法。在多个尺度上调整图像大小，并为每个尺度计算特征图（HOG [8]或深度卷积特征[9]，[1]，[2]）（图1（a））。这种方法通常是有用的但是很费时。第二种方法是在特征图上使用多个尺度（和/或宽高比）的滑动窗口。例如，在DPM [8]中，使用不同的过滤器（卷积）大小（例如5×7和7×5）分别训练不同长宽比的模型。如果使用这种方法处理多个尺度，则可以将其视为“过滤器金字塔”（图1（b））。第二种方法通常与第一种方法结合使用[8]。
相比之下，我们的基于anchor的方法是基于锚点金字塔构建的，这种方法更具成本效益。我们的方法参照多个尺度和宽高比的anchor对边界框进行分类和回归。它仅依赖于单一尺度的图像和特征图，并使用单一大小的过滤器（在特征图上滑动窗口）。我们通过实验证明了该方案对解决多种尺度和size的影响（表8）。
由于基于anchor的这种多尺度设计，我们可以简单地使用在单尺度图像上计算出的卷积特征，就像Fast R-CNN检测器所做的一样[2]。多尺度锚点的设计是共享特征同时无需花费额外成本处理尺度的关键组成部分。
### 3.1.2 损失函数
为了训练RPN，我们为每个anchor分配一个二进制类标签（无论anchor内是不是object）。我们为两种anchor定一个正标签：（i）与真实框具有最高IoU重叠的anchor/anchors，或（ii）与任何一个真实框具有大于0.7的IoU的anchor。请注意，单个真实框可能为多个anchor分配正标签。通常，第二个条件足以确定正样本，但是我们仍然采用第一个条件是因为在极少数情况下，第二个条件可能找不到正样本。如果anchor与所有真实框的IoU值均低于0.3，则为非正例anchor分配负标签。既不是正例也不是负例的anchor对于训练没有帮助。
利用这些定义，我们在Fast R-CNN [2]中将多任务损失之后的目标函数最小化。我们对图像的损失函数定义为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200407190857391.png)
其中，i是mini-batch中的anchor的索引，![pi](https://img-blog.csdnimg.cn/20200407191321829.png)是anchori作为object的预测概率。如果anchor为正，则ground-truth标签![在这里插入图片描述](https://img-blog.csdnimg.cn/20200407191300284.png)为1，如果anchor为负，则ground-truth标签![在这里插入图片描述](https://img-blog.csdnimg.cn/20200407191418561.png)为0。 ![ti](https://img-blog.csdnimg.cn/20200407191445286.png)是代表预测边界框的4个参数化坐标的向量，![ti](https://img-blog.csdnimg.cn/20200407191512656.png)是与正anchor关联的ground-truth的4个参数化坐标。分类损失![Lcls](https://img-blog.csdnimg.cn/20200407191558845.png)是两个类别（object与非object）之间的对数损失。对于回归损失，我们使用![Lreg（ti; t i）= R（ti􀀀t i）](https://img-blog.csdnimg.cn/20200407191752809.png)，其中R是在[2]中定义的稳健损失函数（smooth L1）。![p i Lreg](https://img-blog.csdnimg.cn/20200407191836360.png)表示仅对正anchor![（p i = 1）](https://img-blog.csdnimg.cn/20200407191926717.png)激活回归损失，否则对回归损失禁用![（p i = 0）](https://img-blog.csdnimg.cn/20200407192048852.png)。cls和reg层的输出分别由![fpig](https://img-blog.csdnimg.cn/20200407192150600.png)和![ftig](https://img-blog.csdnimg.cn/20200407192201886.png)组成。
这两项通过Ncls和Nreg归一化，并通过平衡参数加权。在我们当前的实现中（如发布的代码中一样），等式（1）中的cls项通过mini-batch大小（即Ncls = 256）进行归一化，而reg项通过anchor位置的数量（例如，Nreg ：2——400）。默认情况下，我们设置λ= 10，因此cls和reg项的权重大致相等。我们通过实验表明，结果对较大范围内的λ值不敏感（表9）。我们还注意到，上面的标准化不是必需的，可以简化。
对于边界框回归，我们采用[5]中的4个坐标的参数化：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020040719261243.png)
其中x，y，w和h表示框的中心坐标及其宽度和高度。变量x，xa和x*分别用于预测框，anchor  box和真实值框（同样对于y; w; h）。可以将其视为从anchor box到附近的真实值框的边界框回归。
然而，我们的方法通过与以前基于RoI的（感兴趣区域）方法[1]，[2]不同的方式实现边界框回归。在[1]，[2]中，对从任意大小的RoI池化的特征执行边界框回归，并且回归权重由所有区域大小共享。在我们的公式中，用于回归的特征在特征图上具有相同的空间大小（3×3）。为了说明变化的大小，学习了一组k个边界框回归器。每个回归器负责一个比例和一个长宽比，而k个回归器不共享权重。这样，由于anchor的设计，即使特征具有固定的大小/比例，仍然可以预测各种大小的box。
### 3.1.3 训练 RPNs
可以通过反向传播和随机梯度下降（SGD）端对端地训练RPN [35]。我们遵循[2]中的“以图像为中心”的采样策略来训练该网络。每个mini-batch 均处理包含多个正负例anchor的单个图像。可以针对所有anchor的损失函数进行优化，但是会偏向于负样本，因为它们占主导地位。因此，取而代之的是，我们在图像中随机采样256个anchor，以计算mini-batch的损失函数，其中采样的正和负anchor的比例最高为1：1。如果图像中的正样本少于128个，则用负样本填充mini-batch。
我们通过从零均值高斯分布中提取权重（标准偏差为0.01）来随机初始化所有新层。所有其他层（即共享卷积层）都通过预先训练ImageNet分类模型来初始化[36]，这是标准做法[5]。我们调整ZF网络的所有层，并调整conv3_1以及VGG网络以节省内存[2]。对于PASCAL VOC数据集，我们对60k个小批量使用0.001的学习率，对接下来的20k个mini-batch使用0.0001的学习率。我们使用0.9的动量和0.0005的权重衰减[37]。我们的实现使用Caffe [38]。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200407200438363.png)
## 3.2 RPN和Fast R-CNN共享卷积层
到目前为止，我们已经描述了如何训练用于区域提议生成的网络，而没有考虑将利用这些proposal的基于区域的目标检测CNN。对于检测网络，我们采用Fast R-CNN [2]。接下来，我们描述学习具有RPN和Fast R-CNN并具有共享卷积层的统一网络的算法（图2）。
RPN和Fast R-CNN均经过独立训练，将以不同方式修改其卷积层。因此，我们需要开发一种技术，允许在两个网络之间共享卷积层，而不是学习两个单独的网络。我们讨论了三种共享特征的网络训练方法： 
（1）交替训练。在此解决方案中，我们首先训练RPN，然后使用这些proposal来训练Fast R-CNN。然后，使用由Fast R-CNN微调的网络初始化RPN，然后重复此过程。这是本文所有实验中使用的解决方案。
（2）近似联合训练（Approximate joint training）。在此解决方案中，如图2所示，在训练过程中将RPN和Fast R-CNN网络合并为一个网络。在每次SGD迭代中，前向传递都会生成区域proposal，就像Fast R-CNN检测器对待固定的，预先计算的proposal一样。反向传播照常进行，对于共享层，来自RPN Loss和Fast R-CNN Loss的反向传播信号被组合在一起。该解决方案易于实现。但是此解决方案忽略了导数w.r.t. proposal boxes的坐标也是网络响应，因此是近似值。在我们的实验中，我们凭经验发现此求解器产生的结果接近，但与交替训练相比，训练时间减少了约25-50％。这个解决方案包含在我们发布的Python代码中。
（3）非近似联合训练（Non-approximate joint training）。
如上所述，RPN预测的边界框也是输入的函数。Fast R-CNN中的RoI 池化层[2]接受卷积特征，并接受预测的边界框作为输入，因此，理论上有效的反向传播求解器也应包含梯度w.r.t.，框坐标。这些梯度在上面的近似联合训练中被忽略。在一个非近似的联合训练解决方案中，我们需要一个ROI 池化层的可微的（differentiable）w.r.t.框坐标。这是一个不平凡的问题，可以通过[15]中开发的“ RoI warping”层来提供解决方案，这超出了本文的范围。
**4步交替训练。**
在本文中，我们采用务实的4步训练算法通过交替优化来学习共享特征。第一步，我们按照3.1.3节所述训练RPN。该网络使用ImageNet预训练的模型初始化，并针对区域建议任务端到端进行了微调。在第二步中，我们使用步骤1 RPN生成的proposal，通过Fast R-CNN训练一个单独的检测网络。该检测网络也由ImageNet预训练模型初始化。此时，两个网络不共享卷积层。在第三步中，我们使用检测器网络初始化RPN训练，但是我们调整了共享卷积层，并且仅微调了RPN唯一的层。现在，这两个网络共享卷积层。最后，保持共享卷积层固定不变，我们对Fast R-CNN的唯一层进行微调。这样，两个网络共享相同的卷积层并形成统一的网络。可以进行类似的交替训练进行更多迭代，但是我们观察到提升很小。
## 3.3 实现细节
我们在单一尺度的图像上训练和测试区域提议和目标检测网络[1]，[2]。我们重新缩放图像，使它们的短边为s = 600像素[2]。多尺度特征提取（使用图像金字塔）可能会提高准确性，但并不能表现出良好的速度精度折衷[2]。在重新缩放的图像上，最后一个卷积层上的ZF和VGG网络的总跨度为16像素，因此在调整大小（～500×375）之前，在典型的PASCAL图像上为～10像素。即使跨度较大，也可以提供良好的结果，尽管跨度较小时可以进一步提高精度。
对于anchor，我们使用3个尺度，框区域分别为128^2、 256^2和 512^2像素，以及3个宽高比为1：1、1：2和2：1。这些超参数不是为特定数据集精心选择的，我们将在下一部分中提供有关其影响的消融实验。如前所述，我们的解决方案不需要图像金字塔或过滤器（卷积）金字塔即可预测多个尺度的区域，从而节省了可观的运行时间。图3（右）显示了我们的方法在各种尺度和纵横比下的功能。表1显示了使用ZF网络为每个锚点学习的平均建议大小。我们注意到，我们的算法所允许的预测大于潜在的感受野。这样的预测并非没有可能-即使只有object的中间可见，则仍可以粗略地推断出object的范围。
跨越图像边界的anchor需要小心处理。在训练期间，我们将忽略所有跨边界anchor，因此它们不会造成损失。对于典型的1000×600图像，总共将有大约20000（≈60×40×9）个anchor。忽略跨边界anchor，每个图像大约有6000个anchor用于训练。如果在训练中不忽略跨边界的异常值，则会在目标中引入较大且难以校正的误差项，并且训练不会收敛。但是，在测试过程中，我们仍将全卷积RPN应用于整个图像。这可能会生成跨边界建议框，我们会将其裁剪到图像边界。
一些RPN proposal彼此高度重叠。为了减少冗余，我们根据proposal区域的cls分数采用非最大抑制（NMS）。我们将NMS的IoU阈值固定为0.7，这使得每个图像有大约2000个建议区域。正如我们将显示的那样，NMS不会损害最终的检测准确性，但是会大大减少proposal的数量。在NMS之后，我们使用排名前N位的proposal区域进行检测。在下文中，我们使用2000个 RPN proposal训练Fast R-CNN，但是在测试时评估不同数量的proposal。
# 4. 实验
## 4.1 在PASCAL VOC上的实验
我们根据PASCAL VOC 2007检测基准[11]全面评估了我们的方法。该数据集由大约20个物体类别的5k个训练图像和5k个测试图像组成。我们还提供了一些模型的PASCAL VOC 2012基准测试结果。对于ImageNet预训练网络，我们使用具有5个卷积层和3个全连接层的“快速”版本的ZF net [32]，以及具有13个卷积层和3个全连接层的公共VGG-16 模型[3]。我们主要评估检测平均精度（mAP），因为这是目标检测的实际指标（而不是关注目标proposal代理指标）。
表2（顶部）显示了使用各种区域建议方法进行训练和测试时的 Fast R-CNN结果。这些结果使用ZF网络。对于选择性搜索（SS）[4]，我们通过“快速”模式生成了大约2000个proposal。对于EdgeBoxes（EB）[6]，我们通过调整默认EB设置为0.7的IOU来生成proposal。在Fast R-CNN框架下，SS的mAP为58.7％，EB的mAP为58.6％。具有Fast R-CNN的RPN取得了较好的结果，mAP达到59.9％，同时使用了多达300个proposal。由于共享卷积计算，使用RPN产生的检测系统比使用SS或EB的检测系统快得多。较少的提议也降低了区域级全连接层的成本（表5）。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200407225131636.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
**RPN上的消融实验。**
为了研究RPNs作为提议方法的行为，我们进行了一些消融研究。首先，我们展示了在RPN和Fast R-CNN检测网络之间共享卷积层的效果。为此，我们在4步训练过程的第二步之后停止。使用单独的网络会使结果略降至58.7％（RPN + ZF，未共享，表2）。我们观察到这是因为在第三步中，使用检测器调整的特征来微调RPN时，proposal质量得到了改善。
接下来，我们将解开RPN对训练Fast R-CNN检测网络的影响。为此，我们使用2000 SS提案和ZF网络训练了Fast R-CNN模型。我们调整此检测器，并通过更改测试时使用的proposal区域来评估检测mAP。在这些消融实验中，RPN不与检测器共享特征。
在测试时用300个RPN proposal替换SS的mAP为56.8％。 mAP的损失是由于训练/测试proposal之间的不一致。该结果用作以下比较的基准。
令人惊讶的是，在测试时，当使用排名靠前的100个proposal时，RPN仍然可以带来不错的结果（55.1％），表明排名靠前的RPN proposal是准确的。另一方面，使用排名靠前的6000 个RPN proposal（不使用NMS）mAP为55.2％，这表明NMS不会损害检测mAP并可以减少误报。
接下来，我们通过在测试时关闭RPN的cls和reg输出中的一个来分别研究它们的作用。当在测试时删除cls层时（因此不使用NMS /ranking），我们从未计分的区域中随机抽取了N个proposal，当N = 1000时，mAP（55.8％）几乎没有变化，但是当N = 100时，mAP下降至44.6％。这表明cls分数说明了排名最高的proposal的准确性。
另一方面，当在测试时删除reg层（因此proposal成为anchor框）时，mAP下降至52.1％。这表明高质量的proposal主要是由于回归框的边界。尽管anchor框具有多个尺度和宽高比，但不足以进行精确检测。
我们还评估了更强大的网络对RPN proposal质量的影响。我们使用VGG-16训练RPN，但仍使用上述SS + ZF检测器。 mAP从56.8％（使用RPN + ZF）提高到59.2％（使用RPN + VGG）。这是一个令人鼓舞的结果，因为它表明RPN + VGG的proposal质量优于RPN + ZF的proposal质量。由于RPN + ZF的proposal与SS相差无几（当持续用于训练和测试时，两者均为58.7％），因此我们可以预期RPN + VGG会比SS更好。以下实验证明了这一假设。
**VGG-16的性能。**
表3列出了proposal和检测的VGG-16结果。使用RPN + VGG，未共享特征的结果为68.5％，略高于SS基准。如上所示，这是因为RPN + VGG生成的proposal比SS更准确。与预定义的SS不同，RPN受到了积极的训练，并从更好的网络中受益。对于特征共享的变体，结果为69.9％，比强大的SS基准要好，但proposal几乎没有（计算）成本。我们还将在PASCAL VOC 2007和2012的联合训练集上训练RPN和检测网络。mAP为73.2％。图5显示了PASCAL VOC 2007测试集上的一些结果。在PASCAL VOC 2012测试集（表4）上，我们的方法在VOC 2007 trainval + test和VOC 2012 trainval的并集上训练的mAP为70.4％。表6和表7显示了详细的数字。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200408151943704.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200408151858460.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
在表5中，我们总结了整个目标检测系统的运行时间。SS因为内容需要1-2秒（平均约1.5秒），而带有VGG-16的Fast R-CNN在2000个SS提案中需要320毫秒（如果在全连接层上使用SVD则需要223毫秒[2]）。我们的带有VGG-16的系统用于proposal和检测总共需要198毫秒。通过共享卷积特征，仅RPN只需花费10毫秒即可计算附加层。由于更少的proposal（每个图像300个），我们的区域计算也降低了。我们的系统采用ZF网络的帧率为17 fps。
**对超参数的敏感性。**
在表8中，我们调查了anchor的设置。默认情况下，我们使用3个尺度和3个宽高比（表8中的69.9％mAP）。如果在每个位置仅使用一个anchor，则mAP会下降3-4％。如果使用3个尺度（具有1个宽高比）或3个宽高比（具有1个尺度），则mAP更高，表明使用多个尺寸的anchor作为回归参考是有效的解决方案。在此数据集上仅使用3个尺度和1个宽高比（69.8％）相当于使用3个尺度和3个宽高比的，这表明尺度和宽高比不会因检测精度而散乱（disentangled dimensions）。但是我们仍然在设计中采用这两个维度，以保持系统的灵活性。
在表9中，我们比较了公式（1）中的不同值。默认情况下，我们使用λ= 10，这使等式（1）中的两项在归一化后具有大致相等的加权。表9显示，当结果在大约两个数量级（1到100）之间时，我们的结果仅受到少量影响（约为1％）。这表明结果在很大范围内对λ都不敏感。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020040815323990.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
**Recall-to-IoU的分析。**
接下来，我们使用真实框计算不同IoU比率下proposal的召回率。值得注意的是，Recall-to-IoU指标与最终检测精度只是松散的[19]，[20]，[21]。使用此度量标准来诊断proposal方法比评估它更合适。
在图4中，我们显示了使用300、1000和2000个proposal的结果。我们将它们与SS和EB进行比较，根据这些方法所产生的置信度，N个proposal是排名前N位的proposal。这些图表明，当proposal数量从2000个减少到300个时，RPN方法表现得很不错。这解释了为什么当使用最少300个proposal时RPN具有良好的最终检测mAP。正如我们之前分析的那样，此属性主要归因于RPN的cls项。当proposal减少时，SS和EB的召回率比RPN下降得更快。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200408153411756.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
**一阶段检测与两阶段proposal+检测。**
OverFeat论文[9]提出了一种在卷积特征图上的滑动窗口上使用回归器和分类器的检测方法。OverFeat是一阶段的，特定类的检测pipeline，而我们的是一个两阶段的级联，包括与类无关的proposal和特定类的检测。在OverFeat中，区域特征来自尺度金字塔上一个长宽比的滑动窗口。这些特征用于同时确定目标的位置和类别。在RPN中，特征来自正方形（3×3）滑动窗口，并预测相对于具有不同尺度和宽高比的anchor的proposal。尽管两种方法都使用滑动窗口，但是区域提议任务只是Faster RCNN的第一阶段---下游Fast R-CNN检测器会处理proposal以对其进行完善。在我们级联的第二阶段，从proposal框中自适应地池化区域范围的特征[1]，[2]，该proposal框中将更好地覆盖区域的特征。我们相信这些特征可导致更准确的检测。
为了比较一阶段系统和两阶段系统，我们通过一阶段Fast R-CNN模拟了OverFeat系统（因此也避免了实现细节的其他差异）。在此系统中，“proposal”是3个尺度（128、256、512）和3个长宽比（1：1、1：2、2：1）的密集滑动窗口。Fast R-CNN经过训练可以从这些滑动窗口预测特定类别的分数和回归框的位置。由于OverFeat系统采用图像金字塔，因此我们还使用从5个尺度提取的卷积特征进行评估。我们在[1]，[2]中使用这5个尺度。
表10比较了两阶段系统和一阶段系统的两个变体。使用ZF模型，一阶段系统的mAP为53.9％。这比两阶段系统（58.7％）低4.8％。该实验证明了级联区域提议和目标检测的有效性。在[2]，[39]中报告了类似的观察结果，其中用滑动窗口替换SS区域proposal导致两篇论文约6％的退化。我们还注意到，单阶段系统速度较慢，因为它要处理的proposal要多得多。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020040815435314.png)
## 4.2 在MS COCO上的实验
我们在Microsoft COCO目标检测数据集上提供了更多结果[12]。该数据集涉及80个物体类别。我们在训练集上测试了80k张图像，在验证集上测试了40k张图像，在test-dev集上测试了20k张图像。我们评估IoU  [0.5：0.05：0.95]（COCO的标准指标，简称为mAP @ [.5，.95]）和mAP@0.5（PASCAL VOC指标）的平均mAP。
我们针对此数据集对系统进行了一些小的更改。我们在8-GPU实施上训练模型，有效的mini-batch大小对于RPN变为8（每个GPU 1个），对于Fast R-CNN变为16（每个GPU 2个）。RPN步和Fast R-CNN步长均为经过240k次迭代训练，学习率为0.003，然后经过80k次迭代训练为0.0003。我们更改了学习率（从0.003而不是0.001开始），因为更改了mini-batch大小。对于anchor，我们使用3个宽高比和4个尺度（增加64^2），主要是为了处理此数据集上的小目标。此外，在我们的Fast R-CNN步骤中，将负样本定义为用于[1]，[2]中的与真实值具有最大IoU且在[0;  0.5）之间，而不是[0.1;  0.5）之间。我们注意到，在SPPnet系统[1]中，[0.1;  0.5）中的负例用于网络微调，但[0;  0.5）的负例仍在SVM步骤中通过难例挖掘进行访问。但是Fast R-CNN系统[2]放弃了SVM步骤，因此[0;  0.1）从未使用过。包括这些[0;  0.1），对于Fast R-CNN和Faster R-CNN系统，样本在COCO数据集上均提高了mAP@0.5（但对PASCAL VOC的影响可以忽略不计）。
其余的实现细节与PASCAL VOC上的相同。特别是，我们一直使用300个proposal和单尺度（s = 600）测试。在COCO数据集上，每个图像的测试时间仍然约为200毫秒。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200408155445997.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
在表11中，我们首先使用本文中的实现报告了Fast R-CNN系统的结果[2]。我们的Fast R-CNN基线在测试开发集上具有39.3％的mAP@0.5，高于[2]中的报告。我们推测产生这种差距的原因主要是由于负例的定义以及mini-batch大小的变化。我们还注意到，mAP @ [.5，.95]只是可比的（just comparable）。
接下来，我们评估我们的Faster R-CNN系统。使用COCO训练集进行训练，Faster R-CNN在COCO test-dev集上具有42.1％的mAP@0.5和21.5％的mAP @ [.5，.95]。在相同协议下，与Fast RCNN相比，mAP @ 0.5高出2.8％，mAP @ [.5，.95]高出2.2％（表11）。这表明RPN在提高IoU阈值下的定位精度方面表现出色。使用COCO trainval集进行训练，Faster RCNN在COCO test-dev集上具有42.7％的mAP@0.5和21.9％的mAP @ [.5，.95]。图6显示了MS COCO  test-dev集上的一些结果。
**在ILSVRC和COCO 2015竞赛中Faster R-CNN**
我们已经证明，Faster R-CNN受益于更好的特征，这是由于RPN完全学会了通过神经网络来建议区域。即使将深度增加到100层以上，这种观察仍然有效[18]。仅通过用101层残差网（ResNet-101）替换VGG-16 [18]，Faster R-CNN系统就可以在COCO val 集上将mAP从41.5％/ 21.2％（VGG16）提高到48.4％/ 27.2％（ResNet-101）。 He等人提出了与Faster RCNN相关的其他改进。 [18]在COCO test-dev集上获得了55.7％/ 34.9％的单模型结果和59.0％/ 37.4％的整体结果，在COCO 2015目标检测比赛中获得了第一名。同样的系统[18]在ILSVRC 2015目标检测比赛中也获得了第一名，以绝对的优势8.5％超过了第二名。 RPN还是ILSVRC 2015定位和COCO 2015细分比赛第一名获奖作品的基础，有关详细信息，请参见[18]和[15]。
## 4.3 From MS COCO to PASCAL VOC
大规模数据对于改善深度神经网络至关重要。接下来，我们研究MS COCO数据集如何帮助提高PASCAL VOC的检测性能。作为一个简单的基准，我们直接在PASCAL VOC数据集上评估COCO检测模型，而无需对任何PASCAL VOC数据进行微调。该评估是可能的，因为COCO上的类别是PASCAL VOC上的类别的超集。在此实验中，COCO专有的类别将被忽略，并且仅在20个类别以及背景上执行softmax层。在PASCAL VOC 2007测试集上，此设置下的mAP为76.1％（表12）。
即使未使用PASCAL VOC数据，此结果也比在VOC07 ​​+ 12上训练的结果（73.2％）更好。然后，我们在VOC数据集上微调COCO检测模型。在此实验中，COCO模型代替了ImageNet预训练模型（用于初始化网络权重），并且如3.2节中所述对Faster R-CNN系统进行了微调。这样做PASCAL VOC 2007测试集的mAP达到了78.8％。来自COCO数据集的额外数据将mAP提高了5.6％。表6显示，在PASCAL VOC 2007上，使用COCO + VOC训练的模型具有针对每个类别的最佳AP。在PASCAL VOC 2012测试集上也观察到了类似的进步（表12和表7）。我们注意到，获得这些较好结果的测试时间仍然约为每张图像200毫秒。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200408160319298.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200408160340204.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200408160359791.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
# 5. 总结
我们提出了RPN，以高效，准确地生成区域proposal。通过与下游检测网络共享卷积特征，区域proposal步骤几乎是免费的（计算）。我们的方法使基于深度学习的统一目标检测系统能够以接近实时的帧速率运行。所学习的RPN还提高了区域proposal质量，从而提高了总体目标检测精度。
# REFERENCES
[1] K. He, X. Zhang, S. Ren, and J. Sun, “Spatial pyramid pooling in deep convolutional networks for visual recognition,” in European Conference on Computer Vision (ECCV), 2014.
[2] R. Girshick, “Fast R-CNN,” in IEEE International Conference on Computer Vision (ICCV), 2015.
[3] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in International Conference on Learning Representations (ICLR), 2015.
[4] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W. Smeulders, “Selective search for object recognition,” International Journal of Computer Vision (IJCV), 2013.
[5] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2014.
[6] C. L. Zitnick and P. Doll´ar, “Edge boxes: Locating object proposals from edges,” in European Conference on Computer Vision (ECCV), 2014.
[7] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.
[8] P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ramanan, “Object detection with discriminatively trained partbased models,” IEEE Transactions on Pattern Analysis and Machine
Intelligence (TPAMI), 2010.
[9] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun, “Overfeat: Integrated recognition, localization and detection using convolutional networks,” in International Conference on Learning Representations (ICLR), 2014.
[10] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards real-time object detection with region proposal networks,” in Neural Information Processing Systems (NIPS), 2015.
[11] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The PASCAL Visual Object Classes Challenge 2007 (VOC2007) Results,” 2007.
[12] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan,P. Doll´ar, and C. L. Zitnick, “Microsoft COCO: Common Objects in Context,” in European Conference on Computer
Vision (ECCV), 2014.
[13] S. Song and J. Xiao, “Deep sliding shapes for amodal 3d object detection in rgb-d images,” arXiv:1511.02300, 2015.
[14] J. Zhu, X. Chen, and A. L. Yuille, “DeePM: A deep part-based model for object detection and semantic part localization,” arXiv:1511.07131, 2015.
[15] J. Dai, K. He, and J. Sun, “Instance-aware semantic segmentation via multi-task network cascades,” arXiv:1512.04412, 2015.
[16] J. Johnson, A. Karpathy, and L. Fei-Fei, “Densecap: Fully convolutional localization networks for dense captioning,”arXiv:1511.07571, 2015.
[17] D. Kislyuk, Y. Liu, D. Liu, E. Tzeng, and Y. Jing, “Human curation and convnets: Powering item-to-item recommendations on pinterest,” arXiv:1511.04003, 2015.
[18] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” arXiv:1512.03385, 2015.
[19] J. Hosang, R. Benenson, and B. Schiele, “How good are detection proposals, really?” in British Machine Vision Conference (BMVC), 2014.
[20] J. Hosang, R. Benenson, P. Doll´ar, and B. Schiele, “What makes for effective detection proposals?” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2015.
[21] N. Chavali, H. Agrawal, A. Mahendru, and D. Batra, “Object-Proposal Evaluation Protocol is ’Gameable’,” arXiv:1505.05836, 2015.
[22] J. Carreira and C. Sminchisescu, “CPMC: Automatic object segmentation using constrained parametric min-cuts,”IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2012.
[23] P. Arbel´aez, J. Pont-Tuset, J. T. Barron, F. Marques, and J. Malik, “Multiscale combinatorial grouping,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.
[24] B. Alexe, T. Deselaers, and V. Ferrari, “Measuring the objectness of image windows,” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2012.
[25] C. Szegedy, A. Toshev, and D. Erhan, “Deep neural networks for object detection,” in Neural Information Processing Systems (NIPS), 2013.
[26] D. Erhan, C. Szegedy, A. Toshev, and D. Anguelov, “Scalable object detection using deep neural networks,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.
[27] C. Szegedy, S. Reed, D. Erhan, and D. Anguelov, “Scalable,high-quality object detection,” arXiv:1412.1441 (v1), 2015.
[28] P. O. Pinheiro, R. Collobert, and P. Dollar, “Learning to segment object candidates,” in Neural Information Processing Systems (NIPS), 2015.
[29] J. Dai, K. He, and J. Sun, “Convolutional feature masking for joint object and stuff segmentation,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.
[30] S. Ren, K. He, R. Girshick, X. Zhang, and J. Sun, “Object detection networks on convolutional feature maps,”arXiv:1504.06066, 2015.
[31] J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Bengio, “Attention-based models for speech recognition,”in Neural Information Processing Systems (NIPS), 2015.
[32] M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional neural networks,” in European Conference on Computer Vision (ECCV), 2014.
[33] V. Nair and G. E. Hinton, “Rectified linear units improve restricted boltzmann machines,” in International Conference on Machine Learning (ICML), 2010.
[34] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov,D. Erhan, and A. Rabinovich, “Going deeper with convolutions,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2015.
[35] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard,W. Hubbard, and L. D. Jackel, “Backpropagation applied to handwritten zip code recognition,” Neural computation, 1989.
[36] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma,Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg,and L. Fei-Fei, “ImageNet Large Scale Visual Recognition
Challenge,” in International Journal of Computer Vision (IJCV),2015.
[37] A. Krizhevsky, I. Sutskever, and G. Hinton, “Imagenet classificationwith deep convolutional neural networks,” in Neural Information Processing Systems (NIPS), 2012.
[38] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick,S. Guadarrama, and T. Darrell, “Caffe: Convolutional architecture for fast feature embedding,” arXiv:1408.5093, 2014.
[39] K. Lenc and A. Vedaldi, “R-CNN minus R,” in British Machine Vision Conference (BMVC), 2015
