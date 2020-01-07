# 摘要
我们提出了一个单阶段检测框架，该框架解决了多尺度目标检测和类不平衡的问题。我们没有设计更深层的网络，而是引入了一种简单而有效的特征丰富化方案来生成多尺度的上下文特征。我们进一步引入了一种级联的优化（精炼）方案，该方案首先将多尺度的上下文特征注入到一阶段检测器的预测层中，以增强其进行多尺度检测的判别能力。 其次，级联精炼方案通过细化锚（anchors）和丰富的特征以改善分类和回归来解决类不平衡问题。 实验在两个基准上执行：PASCAL VOC和MSCOCO。 对于MS COCO测试上的320×320输入，我们的检测器在单尺度推理的情况下以33.2的COCO AP达到了最先进的一阶段检测精度，操作是在一个Titan  XP GPU上以21毫秒运行的 。 对于MS COCO测试上的512×512输入，与最佳报告的单阶段结果相比，就COCO AP而言，我们的方法获得了一个明显的增加（增加了1.6%）[5]。源代码和模型可在以下位置获得：https://github.com/Ranchentx/EFGRNet.

# 1. 介绍
目标检测是众多现实应用中的一个活跃的研究问题。 基于卷积神经网络（CNN）的现代目标检测方法可分为两类：（1）两阶段方法[33,23]，以及（2）一阶段方法[27,32]。两阶段方法首先生成目标建议，然后对这些建议进行分类和回归。一阶段方法通过输入图像上的规则且密集的采样网格直接定位对象。通常，两阶段目标检测器具有比一阶段方法更准确的优势。另一方面，与两级检测器相比，单级方法具有时间计算效率，但在性能上有所妥协[19]。在这项工作中，我们研究了单阶段框架中的通用对象检测问题。
近年来，已经引入了多种单阶段目标检测方法[27,32,41,24]。 在现有的单级目标检测器中，单发多箱检测器（SSD）[27]最近因其改进的检测性能和高速度的综合优势而广受欢迎。 标准的SSD框架可用于基础网络（例如VGG），并在截断的基础网络的末尾添加一系列卷积层。添加的卷积层和较早的一些基础层具有不同的分辨率， 被用来进行独立的预测。 在标准SSD中，每个预测层都专注于预测特定规模的对象。 它采用金字塔形特征层次结构，其中低层或前一层针对小物体，而深层或高层则针对检测大物体。 在实现高计算效率的同时，SSD在检测精度方面仍落后于大多数现代两级检测器。
在这项工作中，我们区分了标准SSD检测器实现最高精度同时保持其高速度的两个主要障碍。 首先，标准的SSD难以应对大尺度变化[1]。 这可能是由于SSD预测层中的上下文信息固定所致。 现有方法通过例如在更深的骨干网络模型上添加上下文信息[13]和特征金字塔表示[41,24,4,30]来解决该问题。 大多数方法[41,24,4]采用自顶向下的金字塔表示，其中先对深层的低分辨率特征图进行上采样，然后与浅层的高分辨率特征图结合以注入高级语义 信息。 尽管这样的特征金字塔表示有助于解决大尺度变化的问题，但性能仍然远远不能令人满意。
第二个关键问题是在训练SSD检测器期间遇到的前景类-背景类不平衡问题。 该问题的现有解决方案[24,41]包括，例如，在稀疏的难例集上进行训练，同时对经过良好分类的示例对其损失进行打折（down-weights），另外还有整合两阶段anchor优化策略，以通过消除负例的anchors来减少分类器的搜索空间。 尽管取得了成功，但由于这些特征与优化的anchors无法很好地对齐，所以[41]的工作采用了自上而下的特征金字塔表示法，并且仅对anchors进行了优化。 在这项工作中，我们寻求一种替代方法来共同解决多尺度目标检测和类不平衡的问题，从而在不牺牲其高速度的情况下提高SSD的准确性。
## 贡献
**贡献**：我们重新审视了标准的SSD框架，以共同解决多尺度目标检测和类不平衡的问题。 首先，我们引入一种特征增强的方案，以提高标准SSD中预测层的判别能力。 无需使骨干网络模型更深，而是设计了我们的特征增强方案来生成多尺度上下文特征。 我们进一步引入了具有双重对象的级联优化方案。 首先，它将自下而上的金字塔特征层次结构中的多尺度上下文特征注入到标准SSD预测层中。 所得的增强的特征对于尺度变化更鲁棒。 其次，它通过利用增强的特征来执行类不可知的分类和边界框（bounding-box）回归以精确定位，从而解决了类不平衡问题。 然后，进一步利用初始框回归和二元分类来优化相关的增强的特征，​​以获得最终分类分数和边界框回归。
我们对两个具有挑战性的基准（benchmarks）进行了全面的实验：PASCAL VOC 2007 [12]和MSCOCO [25]。 与两个数据集上的现有一阶段方法相比，我们的检测器均能获得更好的结果。 对于512×512的MS COCO测试集，在Titan XP GPU上以39毫秒（ms）的推理时间运行时，在相同主干（VGG）的条件下，我们的检测器性能比RefineDet [41]高4.5％。

# 2. 相关工作
目标检测[33,27,7,28,35]是一个具有挑战性的计算机视觉问题。 基于卷积神经网络[CNN] [36,18,9,38,29,37]的目标检测器[14,15,32,17,33,8,27,2]在 最近几年展示了杰出的性能。 这项工作着重于一阶段目标检测器[32,27]，该检测器通常比其两阶段目标检测器更快。 在现有的单阶段方法中，SSD [27]已显示出可在实时操作时提供出色的性能。 它使用多尺度表示来检测金字塔层次结构中的对象。在这样的层次结构中，浅层有助于预测较小的对象，而较深的层则有助于检测较大的对象。 我们的方法基于标准SSD，因为它具有卓越的准确性和高速度。
一阶段检测器（例如SSD）难以准确地检测出具有明显尺度变化的物体。 此外，SSD检测器还存在类不平衡的问题。 文献[13,3,6,42]中的现有方法通过利用上下文信息，更好的特征提取或自上而下的特征金字塔表示来解决第一个问题。 一种流行的策略是构建自上而下的特征金字塔表示以从较深层向信息受限制的较浅的层注入高级语义信息[24,4]。 [30]的工作提出了一种构造特征金字塔的替代方法，该方法基于称为特征图像金字塔的图像金字塔。 相反，我们的方法不需要任何特征化的图像金字塔或自上而下的金字塔结构，而是专注于捕获多尺度上下文信息。 而且，我们的方法包括一个专门的模块来解决类不平衡问题。 文献[6]的工作是通过多变形（multi-deformable)的头部来研究上下文的整合，并使用框（box）回归（位置和比例偏移）来优化特征。 相反，我们以两种方式提高了标准SSD预测层的判别能力。 首先，我们从多分支ResNeXT架构[39,31]中引入了一种特征增强的方案，该方案产生了多尺度的上下文特征，以利用上下文信息增强标准的SSD特征。 其次，我们引入了级联的优化方案，在这种方案中，同时使用了框回归和二元（binary）分类来优化特征。 二元分类（目标类别预测）用于生成突出显示可疑对象位置的对象性（objecness）图。 在特征优化期间，仅位置偏移用于特征与优化（anchors）的对齐，而比例偏移则被忽略。
为了解决训练阶段类别不平衡的问题，RetinaNet [24]引入了focal loss来降低简单样本的贡献。 RefineDet [41]提出了一个两步anchor优化模块，通过删除几个负anchors来减少分类器的搜索空间。 另外，anchor优化模块粗略地调整anchor的位置。 与[41]不同，我们的级联优化方案通过首先将多尺度上下文信息注入到标准SSD预测层中来利用增强的特征。 此外，级联优化去除了几个负anchors ，不仅细化了anchor的位置，还细化了特征。

# 3. 方法
我们的检测框架由三部分组成：标准SSD层，特征增强（丰富）（FE）方案和级联优化方案。 我们的FE方案（第3.1节）包含一个多尺度上下文特征模块（MSCF）以解决尺度变化。FE方案产生了多尺度的上下文特征，以提高标准SSD预测层的判别能力。级联优化方案（第3.2节）同时利用了多尺度上下文和标准SSD特征，并解决了类不平衡问题。级联优化方案通过分别在两个级联模块（即物体性模块（OM）和特征导向的优化模块（FGRM））中执行框回归和分类来优化anchor和特征。物体性模块（OM）对对象与背景进行二进制分类，并进行初始框回归。然后，FGRM模块重新优化特征和anchor位置，以预测最终的多类别分类和边界框位置。![图1](https://img-blog.csdnimg.cn/20191129203400138.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
  **图1：**（a）使用VGG主干的单阶段检测方法的总体架构。它由三部分组成：标准SSD层，特征增强方案和级联优化方案。特征增强方案设计为使用（b）中所示的MSCF模块提取多尺度上下文特征。然后，将这些上下文特征注入到SSD预测层（conv4 3）中，并使用自下而上的特征层次结构在级联优化方案的对象性模块中进一步传播。对象性模块还执行类未知的分类（C1x）和初始回归（B1x）。此外，类未知的分类提供了稍后在我们的级联优化方案的（c）中所示的FGRM模块中使用的对象性图 。 FGRM模块生成用于预测最终分类（C2x）和边界框回归（B2x）的最终优化的特征。
如图1所示，图1说明了使用VGG作为骨干网时我们框架的整体架构。
 根据[41]，我们仅利用四个预测层（conv4 3，fc7，conv8 2，conv9 2）进行检测，而不是原始SSD中使用的六个预测层。将预测层增加到四个以上不会改善我们的性能。

 ## 3.1 特征丰富方案
在标准SSD框架中，特征的提取是从深度卷积网络主干（例如 VGG16或ResNet）中通过卷积和最大池操作的重复过程执行的。尽管保留了一定程度的语义信息，但它们仍然丢失了可能有助于区分对象区域和背景区域的低级特征信息。此外，在每个预测层的恒定感受野仅捕获固定的上下文信息。在这项工作中，我们引入了一种特征增强（FE）方案来捕获多尺度上下文信息。我们首先通过简单的池化操作对输入图像进行下采样，以使其尺寸与第一个SSD预测层的尺寸相匹配。然后，将经过下采样的图像通过我们的多尺度上下文特征（MSCF）模块。
**多尺度上下文特征模块**：提出的MSCF模块在图1（b）中以蓝色虚线框突出显示。它是一个简单的模块，包含多个卷积运算，并产生多尺度的上下文特征。 MSCF模块的结构受到多分支ResNeXT体系结构的启发[39，31]，是拆分，转换和聚合策略的一种操作。MSCF模块将下采样后的图像作为输入，并输出上下文增强的多尺度特征。下采样的图像首先通过大小为3×3和1×1的两个连续卷积层，从而产生初始特征投影。然后，将这些特征投影通过1×1卷积层切成三个低维分支。为了捕获多尺度上下文信息，我们对不同的分支采用三个膨胀卷积[40]，膨胀率分别设置为1、2和4。膨胀卷积的运算将初始特征投影转换为上下文增强的特征集。然后，这些变换后的特征通过级联运算进行聚合，然后传递给1×1卷积进行运算。 MSCF的输出用于我们的级联优化方案的对象性模块（OM）中。
## 3.2 级联优（细）化方案
我们的优化方案由两个级联模块组成：对象性模块和特征导向的优化模块（FGRM），如图1（a）所示。对象性模块通过多尺度上下文信息增强了SSD的特征，并标识了可能的物体位置（客观性）。使用多尺度上下文信息增强特征可以提高对于小目标的性能，而FGRM使用对象性预测来解决类不平衡问题。
**物体性模块**：物体性模块首先通过逐元素乘法运算在conv4_3的MCSF模块中注入多尺度上下文信息，从而增强了SSD的特征。 然后，我们引入了一个自下而上的金字塔特征层次结构，以将增强的特征传播到后续的SSD预测层，如图1（a）所示。 物体性模块使用步长为2（D）的3×3卷积运算，并投影前一层的特征以与当前层的空间分辨率和通道数匹配。 然后，通过在每个预测层上的投影特征和SSD特征之间执行逐元素乘法来获得增强的特征。 最后，增强的特征用于在每个预测层x上执行二元分类（C1x）和初始框回归（B1x）。 x = 1,2、3和4对应于四个预测层。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129204812924.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70#pic_center)
**图2：** 来自PASCAL VOC数据集的示例图像以及来自标准SSD（第二列），多尺度上下文特征（第三列）和增强的特征（第四列）的相应fc7层特征图。 这些示例表明，通过将多尺度上下文特征注入到标准SSD特征中而获得的增强的特征有助于更好地从背景中区分对象区域。

图2显示了来自PASCAL VOC数据集的示例图像，以及来自标准SSD（第二列），D之后的多尺度上下文特征（第三列）和增强的特征（第四列）的相应fc7层特征图。 这些示例表明，利用多尺度上下文信息来增强标准SSD的特征有助于更加关注包含对象实例的区域。 从物体性模块输出的二元分类C1x在FGRM中进一步用于通过过滤掉大量的负anchor来减少正负anchors之间的类不平衡。 另外，C1x输出用于生成注意力图，以引导增强的特征在抑制背景的同时更加关注目标对象。 FGRM中还使用了box回归B1x输出来优化特征和anchors的位置。
	**特征导向的优化模块**：物体性模块中的二元分类器（C1x）输出将每个anchor预测为对象/背景，用于生成突出显示可能的对象位置的物体性图O1x。我们在给定空间位置的所有anchors的目标类别预测上沿着通道轴执行最大池化操作，然后进行Sigmod激活。结果，产生了空间物体性图O1x，该空间物体性图用于改善从物体性模块获得的增强特征Fin：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019112920513255.png#pic_center)
⊙是逐元素乘法，Fm是经过改进后的增强特征。
*内核偏移提取*：物体性和FGRM模块的框回归预测了四个输出：△x，△y，△h和△w。前两个（△x，△y）对应于空间偏移，后两个（△w，△h）对应于空间尺寸的比例偏移。在这里，我们使用来自物体性模块的空间偏移量（x，y），通过将内核偏移量pk估算为下图所示，来指导FGRM中的特征优化：
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/2019112920525160.png#pic_center)
 其中，f1×1表示卷积层，其内核大小为1×1，而B1x  △x，△y表示由对象性模块预测的空间偏移量（△x，△y）。最后，内核偏移量用作可变形卷积的输入[11]，以引导特征采样并与精化的anchors对齐。

*局部上下文信息*：为了在给定的空间位置进一步增强上下文信息，我们在FGRM中利用膨胀卷积[40]。我们分别在步幅为8、16、32、64的SSD预测层上将膨胀率设置为5、4、3和2。
总之，在FGRM中进行所有操作后获得的最终优化特征Frf表示为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019112920542837.png#pic_center)
其中p0表示最终优化特征图Frf中的每个空间位置，d为膨胀率。 R是用于采样输入特征的规则网格（即，如果内核为3×3，膨胀率为1，R =（-1，-1），（-1，0），...，（0，1）， （1，1））。最终的改进特征Frf是由w加权的采样值的总和。 pk是内核偏移量，用于扩展常规采样网格，从而增强CNN模型几何变换的能力。通常，在可变形卷积中，通过在同一输入特征图上应用卷积层来获得偏移。在我们的FGRM中，偏移量是由对象性模块的第一个box回归生成的。为了获得精确的anchor位置，我们遵循与[41]中类似的策略。我们利用从对象性模块预测的偏移量（B1x）来优化原始anchor位置。因此，优化的位置和优化的特征Frf用于执行多类分类（C2x）和框回归（B2x）。
# 4. 实验
## 4.1 数据集和评估指标
**数据集**：我们在两个基准上进行实验：PASCAL VOC 2007 [12]和MS COCO [25]。 PASCAL VOC 2007数据集包含20个不同的物体类别。
 我们对具有5k个图像的VOC 2007 trainval和具有11k个图像的VOC 2012 trainval的组合集进行训练，其中对具有5k个图像的VOC 2007测试集进行评估。 MS COCO是具有80个物体类别的更具挑战性的数据集，分为80k训练，40k验证和20k测试-验证图像。训练在trainval35k集合上进行，评估在minival集合和test-dev2015上进行。
**评估指标**：我们遵循最初用两个数据集定义的评估标准协议。对于Pascal VOC，以均值平均精度（mAP）的形式报告结果，该均值是在交并比（IOU）重叠超过阈值0.5时测到的。 MS COCO的评估指标与Pascal VOC不同，Pascal VOC的总体性能（平均精度（AP））是通过对多个IOU阈值（0.5到0.95）进行平均来测量的。
## 4.2 实验细节
我们的框架采用在ImageNet [34]上进行预训练的VGG-16作为主干架构。我们对两个数据集使用相同的设置进行模型初始化和优化。采用warming up策略，将前5个epochs的初始学习速率设置为从10-6到4×10-3。然后，对于150和200 epoch的PASCAL VOC 2007数据集以及90、120和140 epoch的MS COCO数据集，我们将学习率逐渐降低10倍。对于这两个数据集，权重衰减设置为0.0005，动量设置为0.9，批大小（batch size）为32。在我们的实验中，分别为PASCAL VOC 2007和MS COCO数据集执行了250和160个epoch。除了VGG-16，我们还对MS COCO数据集使用更强大的ResNet-101主干进行了实验。对于ResNet-101，在截断的ResNet-101主干的末尾添加了两个额外的卷积层（即res6 1，res6 2）。我们利用四个预测层（res3，res4，res5，res6 2）进行检测。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129205752825.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70#pic_center)
**表1：** 我们的方法与PASCAL VOC 2007测试集上现有最优的检测器比较。对于300×300和512×512输入，我们的检测器性能优于现有的一阶段方法。
## 4.3 与最先进的技术方法比较
**PASCAL VOC 2007**：在这里，我们将我们的方法与文献中最新的单阶段和两阶段目标检测方法进行比较。表1显示了PASCAL VOC 2007测试集的结果。请注意，大多数现有的两阶段方法都依靠较大的输入图像大小（通常为1000×800）来提高性能。在现有的两阶段目标检测器中，CoupleNet [45]获得的检测分数为82.7 mAP。在单阶段方法的情况下，我们使用两个输入变量进行比较：300×300和〜500×500范围。在输入图像尺寸为300×300的情况下，基准SSD方法的检测精度为77.2 mAP。我们的检测器相对于基准SSD提供了基于mAP的4.1％的显著的绝对的增益。在512×512的输入图像尺寸下，RefineDet [41]和RFBNet [26]的mAP精度分别达到81.8和82.1。在相同的输入大小和主干网络条件下，我们的方法在此数据集上的性能优于RFBNet [26]，准确度为82.7 mAP。图3显示了使用我们的检测器的PASCAL VOC 2007测试集的结果。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129210032697.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
**图3：** 我们对VOC 2007测试集的方法的定性结果（对应于82.7 mAP）。每种颜色都属于一个对象类。

**MS COCO**：表2显示了与SOTA的比较。输入尺寸为320×320时，基准SSD的总体检测得分为25.1。与使用相同主干的基准SSD相比，我们的方法在总体检测得分方面获得了8.1％的显著提高。值得注意的是，与基准SSD相比，中型和小型物体分别实现了11.2％和6.8％的大幅增长。在现有的单阶段方法中，RFBNet [26]和EFIPNet [30]在300×300输入的情况下分别提供30.3和30.0的整体检测精度。我们的方法使用近似相似的输入比例（320×320）和相同的主干网络，得到了一个总检测得分为33.2的SOTA。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019112921044912.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70#pic_center)
*：在Pytorch 0.4.1中使用单个NVIDIA Titan X PASCAL和批大小1进行了测试，以进行公平比较。
**表2：** MS COCO test-dev2015的与SOTA的比较。对于300×300的输入，我们的方法在不显着降低速度的情况下优于现有的单阶段方法。对于512×512输入，CornerNet提供了最佳的整体检测精度。但是，我们的检测器提供的速度比CornerNet快5倍，而IoU阈值为0.5时，其精度比CornerNet更高。我们还将我们的方法的多尺度推理（MS）变体与最新方法（相应论文中的数字）进行了比较。

输入尺寸为512×512和VGG主干，基准SSD的整体检测得分为28.8。我们的方法在输入大小和主干相同的情况下，总体检测精度达到37.5，大大优于基准SSD。当使用功能更强大的ResNet-101主干（总检测得分为39.0）时，我们的检测器可进一步提高性能。当使用512×512输入时，CornerNet [22]以40.6的AP得分达到最佳的整体检测精度。我们的方法提供了比CornerNet [22]高5倍的速度，同时在IoU阈值为0.5时精度更高。 ExtremeNet [43]和CornerNet [22]在更高的IoU（在总AP中有所体现）方面均表现出色，这可能是由于计算上昂贵的多尺度沙漏（Hourglass）架构。图4示出了在coco test-dev上的检测结果。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129210723353.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
**图4：** 我们的检测器在MS COCO 2015 test-dev上的定性检测结果。检测结果对应于37.5 AP。

我们使用[25]提供的分析工具对MS COCO进行了错误分析。图5显示了在所有COCO类别中输入为320×320的RefineDet [41]（左）与我们的方法（右）的比较。 IoU = 0.75时，RefineDet的整体性能为0.309，完美的定位可能会将AP提升至0.583。同样，消除背景假阳性会使结果增加到0.841 AP。 IoU = 0.75时，我们的检测器的整体性能为0.349，完美的定位可能会使AP增至0.611。同样，消除背景假阳性将使性能提高到0.846 AP。我们的方法显示出优于RefineDet的性能。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129210903734.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70#pic_center)
**图5：** 对于所有80个COCO对象类别，在RefineDet [41]（左侧）和我们的检测器（右侧）之间进行了误差分析。为了公平比较，两种方法都使用相同的主干（VGG）和输入大小（320×320）进行分析。在此，每个子图像中的图都显示了一系列精确的召回曲线。这些曲线是使用不同的设置计算的[25]。此外，图例中还显示了AUC曲线。

## 4.4 基准比较
我们首先将特征增强（第3.1节）和级联优化（第3.2节）方案集成到基准SSD中，以评估它们的影响。表3显示了PASCAL VOC 2007和MS COCO数据集的结果。为了公平比较，我们对所有实验都使用相同的设置。在PASCAL VOC 2007数据集上，基准SSD达到77.2 mAP。特征增强方案的引入使mAP较基准SSD提高了2.2％。请注意，特征增强方案是通过对象性模块集成到基准SSD中的。通过级联优化方案的集成，检测性能从77.2 mAP提高到81.0 mAP。为了公平地评估我们的级联优化，我们排除了对象性模块的特征增强和对象性模块中的自下而上的特征层次。特征增强和级联优化方案组合一起提供了比基准SSD高出4.2％的mAP增益。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019112921115898.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70#pic_center)
**表3：** 比较在PASCAL VOC 2007和MS COCO minival set数据集上将我们提出的特征增强和级联的优化方案集成到基准SSD框架中的比较。对于所有实验，主干为VGG16，输入为320×320。我们的最终方法在两个数据集上都比基线SSD的性能有大幅度提高。

在MS COCO数据集上，基准SSD的总体精度为24.4 AP。我们特征增强方案的引入将AP的整体性能从24.4显着提高到29.1。在中等大小的物体上获得了显着的精度提高。集成我们的级联优化方案可将AP中基准SSD的整体精度从24.4提高到31.1。在小型物体上可获得显著的性能提升。我们的最终框架结合了特征增强和级联优化方案，可提供33.0 AP的整体精度，比基准SSD高出8.6％。

**PASCAL VOC 2007的消融研究**：我们在特征增强方案中尝试了三种不同的MSCF模块设计。表4显示了使用三个不同的膨胀率（即1、2、4）的分支时的结果。当在我们的MSCF中使用三个分支时，可获得79.4 mAP的最佳结果，突出了捕获多尺度上下文信息的重要性。我们进一步研究了添加具有不同膨胀率的其他分支。但是，这不会带来任何性能改进。接下来，我们在级联优化方案中分析特征导向优化模块（FGRM）中内核偏移的影响。表5显示了在FGRM的可变形卷积计算中使用不同类型的偏移生成时的比较。我们还报告了标准的膨胀卷积结果（80.2 mAP）。在标准可变形卷积（第二行）的情况下，使用卷积层来学习偏移量[11]。一种直接的方法是通过将偏移量直接应用于标准特征Fm来学习偏移量。与标准的膨胀卷积相比，这表明性能略有提高。来自对象性模块B1x的初始box回归预测位置和比例偏移量（△x，△y，△h，△w），可用于通过1×1卷积学习偏移量。仅使用比例偏移（△h，△w）会降低性能。当使用位置偏移量（△x，△y）生成可变形卷积的偏移量时，可获得最佳结果-81.0 mAP。在整个实验中，我们使用与3.2节相同的膨胀率。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129211423271.png#pic_center)
**表4：** 在Pascal VOC2007测试集的特征增强方案中有关MSCF模块设计的消融实验。结果表明，使用多尺度上下文信息可以提高检测性能。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129211500691.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70#pic_center)
**表5：** 在PASCAL VOC 2007上 ，在我们的FGRM的可变形卷积运算中使用不同类型的偏移生成的性能比较。如[11]中所产生的偏移仅比膨胀卷积在性能上有微小的提高。来自对象性模块B1x的初始box回归可预测位置和比例偏移（△x，△y，△h，△w）。当使用位置偏移量（△x，△y）生成可变形卷积的偏移量时，可获得最佳结果。

# 5. 总结
我们提出一种单阶段方法，共同解决多尺度检测和类别不平衡的问题。我们介绍了一种特征增强方案，以产生多尺度上下文特征。此外，我们提出了一种级联优化方案，该方案首先将这些上下文特征注入到SSD特征中。其次，它利用增强的特征来执行与类无关的分类和边界框回归。然后，使用初始框回归和二元分类来优化特征，然后将其用于获得最终分类分数和边界框回归。在两个数据集上进行的实验表明，我们的方法优于现有的单阶段方法。
# 6.参考文献
[1] Yancheng Bai, Yongqiang Zhang, Mingli Ding, and Bernard Ghanem. Sod-mtgan: Small object detection via multi-task generative adversarial network. In Proc. European Confer-ence on Computer Vision, 2018. 1
[2] Zhaowei Cai and Nuno Vasconcelos. Cascade r-cnn: Delving into high quality object detection. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2018. 2
[3] Guimei Cao, Xuemei Xie,Wenzhe Yang, Quan Liao, Guangming Shi, and JinjianWu. Feature-fused SSD: Fast detection for small objects. arXiv preprint arXiv:1709.05054, 2017. 2
[4] Jiale Cao, Yanwei Pang, and Xuelong Li. Triply supervised decoder networks for joint detection and segmentation.In Proc. IEEE Conference on Computer Vision and Pattern Recognition, June 2019. 1, 2, 6
[5] Kean Chen, Jianguo Li, Weiyao Lin, John See, Ji Wang,Lingyu Duan, Zhibo Chen, Changwei He, and Junni Zou.Towards accurate one-stage object detection with ap-loss.In Proc. IEEE Conference on Computer Vision and Pattern Recognition, June 2019. 1, 6
[6] Xingyu Chen, Junzhi Yu, Shihan Kong, Zhengxing Wu,and Li Wen. Dual refinement networks for accurate and fast object detection in real-world scenes. arXiv preprint arXiv:1807.08638, 2018. 2, 6
[7] Hisham Cholakkal, Jubin Johnson, and Deepu Rajan. Backtracking scspm image classifier for weakly supervised topdown saliency. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, June 2016. 2
[8] Hisham Cholakkal, Jubin Johnson, and Deepu Rajan. Backtracking spatial pyramid pooling-based image classifier for weakly supervised top-down salient object detection.IEEE Transactions on Image Processing, 27(12):6064–6078,2018. 2
[9] Hisham Cholakkal, Guolei Sun, Fahad Shahbaz Khan, and Ling Shao. Object counting and instance segmentation with image-level supervision. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, June 2019. 2
[10] Jifeng Dai, Yi Li, Kaiming He, and Jian Sun. R-fcn: Object detection via region-based fully convolutional networks. In Proc. Advances in Neural Information Processing Systems,2016. 5
[11] Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei. Deformable convolutional networks. In Proc. IEEE international conference on computer vision, pages 764–773, 2017. 4, 8
[12] Mark Everingham, Luc Van Gool, Christopher KI Williams,JohnWinn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International Journal of Computer Vision, 88(2):303–338, 2010. 2, 5
[13] Cheng-Yang Fu, Wei Liu, Ananth Ranga, Ambrish Tyagi,and Alexander C Berg. Dssd: Deconvolutional single shot detector. arXiv:1701.06659, 2017. 1, 2, 5, 6
[14] Ross Girshick. Fast R-CNN. In Proc. IEEE International Conference on Computer Vision, 2015. 2
[15] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2014. 2
[16] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In Proc. IEEE International Conference on Computer Vision, 2017. 6
[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.Spatial pyramid pooling in deep convolutional networks for visual recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015. 2
[18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.Deep residual learning for image recognition. In Proc. IEEE Conference on Computer Vision and Pattern Recognition,2016. 2, 5
[19] Jonathan Huang, Vivek Rathod, Chen Sun, Menglong Zhu,Anoop Korattikara, Alireza Fathi, Ian Fischer, ZbigniewWojna,Yang Song, Sergio Guadarrama, and Kevin Murphy.Speed/accuracy trade-offs for modern convolutional object detectors. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2017. 1
[20] Tao Kong, Fuchun Sun, Chuanqi Tan, Huaping Liu, and Wenbing Huang. Deep feature pyramid reconfiguration for object detection. In Proc. European Conference on Computer Vision, 2018. 5
[21] Tao Kong, Fuchun Sun, Anbang Yao, Huaping Liu, Ming Lu,and Yurong Chen. Ron: Reverse connection with objectness prior networks for object detection. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2017. 5
[22] Hei Law and Jia Deng. Cornernet: Detecting objects as paired keypoints. In The European Conference on Computer Vision, September 2018. 6, 7
[23] Tsung-Yi Lin, Piotr Doll´ar, Ross B Girshick, Kaiming He,Bharath Hariharan, and Serge J Belongie. Feature pyramid networks for object detection. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2017. 1
[24] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection. In Proc.IEEE International Conference on Computer Vision, 2017.1, 2, 6
[25] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Proc. European Conference on Computer Vision, 2014. 2, 5,7
[26] Songtao Liu, Di Huang, and andYunhong Wang. Receptive field block net for accurate and fast object detection. In Proc.European Conference on Computer Vision, 2018. 5, 6
[27] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C Berg. Ssd: Single shot multibox detector. In Proc. European Conference on Computer Vision, pages 21–37, 2016. 1,2, 3, 5, 6
[28] Yanwei Pang, Jiale Cao, and Xuelong Li. Cascade learning by optimally partitioning. IEEE Transactions on Cybernetics, 47(12):4148–4161, 2017. 2
[29] Yanwei Pang, Manli Sun, Xiaoheng Jiang, and Xuelong Li. Convolution in convolution for network in network.IEEE Transactions on Neural Networks Learning Systems,29(5):1587–1597, 2018. 29545
[30] Yanwei Pang, Tiancai Wang, Rao Muhammad Anwer, Fahad Shahbaz Khan, and Ling Shao. Efficient featurized image pyramid network for single shot detector. In Proc. IEEE Conference on Computer Vision and Pattern Recognition,June 2019. 1, 2, 5, 6
[31] Yanwei Pang, Jin Xie, and Xuelong Li. Visual haze removal by a unified generative adversarial network. IEEE Transactions on Circuits and Systems for Video Technology, 2018. 2,3
[32] Joseph Redmon and Ali Farhadi. Yolo9000: Better, faster,stronger. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2017. 1, 2
[33] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.Faster r-cnn: Towards real-time object detection with region proposal networks. In Proc. Advances in Neural Information Processing Systems, 2015. 1, 2, 6
[34] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,Aditya Khosla, Michael Bernstein, and L. Fei-Fei. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115(3):211–252, 2015. 5
[35] Fahad Shahbaz Khan, Jiaolong Xu, Joost van de Weijer,Andrew Bagdanov, Rao Muhammad Anwer, and Antonio Lopez. Recognizing actions through action-specific person detection. IEEE Transactions on Image Processing,24(11):4422–4432, 2015. 2
[36] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In Proc.Advances in Neural Information Processing Systems, 2014.2
[37] Hanqing Sun and Yanwei Pang. Glancenets: efficient convolutional neural networks with adaptive hard example mining.Science China Information Sciences, 61(10):109–101, 2018.2
[38] WenguanWang, Shuyang Zhao, Jianbing Shen, Steven C. H.Hoi, and Ali Borji. Salient object detection with pyramid attention and salient edges. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, June 2019. 2
[39] Saining Xie, Ross Girshick, Piotr Doll´ar, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017. 2, 3
[40] Fisher Yu and Vladlen Koltun. Multi-scale context aggregation by dilated convolutions. arXiv preprint
arXiv:1511.07122, 2015. 3, 4
[41] Shifeng Zhang, Longyin Wen, Xiao Bian, Zhen Lei, and Stan Z. Li. Single-shot refinement neural network for object detection. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2018. 1, 2, 3, 5, 6, 7
[42] Zhishuai Zhang, Siyuan Qiao, Cihang Xie, Wei Shen, Bo Wang, and Alan L. Yuille. Single-shot object detection with enriched semantics. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2018. 2, 5, 6
[43] Xingyi Zhou, Jiacheng Zhuo, and Philipp Krahenbuhl. Bottom-up object detection by grouping extreme and center points. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2019. 6, 7
[44] Chenchen Zhu, Yihui He, and Marios Savvides. Feature selective anchor-free module for single-shot object detection.In Proc. IEEE Conference on Computer Vision and Pattern Recognition, June 2019. 6
[45] Yousong Zhu, Chaoyang Zhao, Jinqiao Wang, Xu Zhao, Yi Wu, and Hanqing Lu. Couplenet: Coupling global structure with local parts for object detection. In Proc. IEEE International Conference on Computer Vision, Oct 2017. 5, 69546






