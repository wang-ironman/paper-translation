# 摘要

我们推出了一个新的目标检测方法---YOLO。先前有关目标检测的工作将分类器用于执行检测。取而代之的是，我们将目标检测框架化为空间分隔的边界框和相关类概率的回归问题。单个神经网络可以在一次评估中直接从完整图像中预测边界框和类概率。 由于整个检测pipeline是单个网络，因此可以直接在检测性能上进行端到端优化。

我们的统一体系结构非常快。 我们的基础YOLO模型以每秒45帧的速度实时处理图像。 较小的网络Fast YOLO每秒可处理惊人的155帧，同时仍实现其他实时检测器的mAP两倍的效果。与最新的检测系统相比，YOLO会产生更多的定位错误，但预测背景假阳性的可能性较小。 最后，YOLO学习了非常普通的目标表示形式。 从自然图像推广到艺术品等其他领域时，它的性能优于其他检测方法，包括DPM和R-CNN。

# 1. 引言                                               

人们看了一眼图像，立即知道图像中有什么物体，它们在哪里以及它们如何相互作用。人类的视觉系统快速准确，使我们能够执行一些复杂的任务，例如在没有意识的情况下驾驶。 快速，准确的物体检测算法将允许计算机在没有专用传感器的情况下驾驶汽车，使辅助设备向人类用户传达实时场景信息，并释放通用响应型机器人系统的潜力。

当前的检测系统重新利用分类器来执行检测。 为了检测物体，这些系统采用了该物体的分类器，并在测试图像的各个位置和比例上对其进行了评估。 像可变形零件模型（DPM）之类的系统使用滑动窗口方法，其中分类器在整个图像上均匀分布的位置上运行[10]。                                                                                                                  

R-CNN等最新方法使用区域提议方法，首先在图像中生成潜在的边界框，然后在这些提议的框上运行分类器。 分类后，使用后处理来完善边界框，消除重复检测并根据场景中的其他目标对这些框进行重新评分[13]。 这些复杂的pipeline运行缓慢且难以优化，因为每个单独的组件都必须分别进行训练。

我们将目标检测重新构造为一个回归问题，直接从图像像素到边界框坐标和类概率。 使用我们的系统，您只需看一次（YOLO）图像即可预测存在的物体及其位置。

![image-20200408224703115](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200408224703115.png)

YOLO非常简单：请参见图1。单个卷积网络可同时预测多个边界框和这些框的类概率。YOLO训练完整图像并直接优化检测性能。 与传统的目标检测方法相比，此统一模型具有多个优点。

首先，YOLO非常快。 由于我们将检测框架化为回归问题，因此不需要复杂的流程。
我们只需在测试时在新图像上运行神经网络即可预测检测结果。 我们的基本网络以每秒45帧的速度运行，在Titan X GPU上没有批处理，而快速版本的运行速度超过150 fps。 这意味着我们可以以不到25毫秒的延迟实时处理流视频。 此外，YOLO达到了其他实时系统平均精度的两倍以上。 有关在网络摄像头上实时运行的系统的演示，请参阅我们的项目网页：http://pjreddie.com/yolo/.

其次，YOLO在2016年5月9日1 arXiv：1506.02640v5 [cs.CV]进行预测时会对图像进行全局预测。 与基于滑动窗口和区域提议的技术不同，YOLO在训练和测试期间会看到整个图像，因此它隐式地编码有关类及其外观的上下文信息。Faster R-CNN是一种顶部检测方法[14]，它会将图像中的背景色块误认为是目标，因为它看不到更大的上下文。 与Fast R-CNN相比，YOLO产生的背景错误少于一半。

第三，YOLO学习目标的可概括表示。在自然图像上进行训练并在艺术品上进行测试时，YOLO在很大程度上优于DPM和R-CNN等顶级检测方法。 由于YOLO具有高度通用性，因此在应用于新域或意外输入时，分解的可能性较小。

YOLO在准确性方面仍落后于最新的检测系统。 尽管它可以快速识别图像中的物体，但仍难以精确定位某些目标，尤其是小的目标。 我们在实验中进一步研究了这些折衷。

我们所有的训练和测试代码都是开源的。 各种预训练的模型也可以下载。

# 2.统一检测

我们将目标检测的各个组成部分统一为一个神经网络。 我们的网络使用整个图像中的特征来预测每个边界框。 它还可以同时预测图像所有类的所有边界框。 这意味着我们的网络会全局考虑整个图像和图像中的所有目标。YOLO设计可实现端到端的训练和实时速度，同时保持较高的平均精度。

我们的系统将输入图像划分为S×S网格。如果物体的中心落入网格单元，则该网格单元负责检测该物体。每个网格单元预测B边界框和这些框的置信度得分。 这些置信度得分反映了该模型对box包含一个物体的信心，以及它认为box预测的准确性。 形式上，我们将置信度定义为Pr（Object）* IOU（truth ，pred）。 如果该单元格中没有物体，则置信度分数应为零。 否则，我们希望置信度分数等于预测框与真实框之间的交并比（IOU）。

每个边界框由5个预测组成：x，y，w，h和置信度。（x，y）坐标表示框相对于网格单元边界的中心。 w和h是相对于整个图像预测的宽度和高度。 最后，置信度预测表示预测框与任何真实框之间的IOU。

每个网格单元还预测C个条件类概率Pr（Classi|Object）。 这些概率以包含目标的网格单元为条件。 无论框B的数量如何，我们仅预测每个网格单元的一组类概率。

在测试时，我们将条件类概率与各个框的置信度预测相乘，![image-20200408230226607](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200408230226607.png)，这为我们提供了每个框的特定类的置信度得分 。 这些分数既编码了该类别出现在box中的概率，也代表了预测box符合这个物体的程度。

![image-20200408230430971](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200408230430971.png)

为了评估PASCAL VOC上的YOLO，我们使用S = 7，B =2。PASCAL VOC有20个标记的类，因此C = 20。我们的最终预测是7×7×30的张量。

## 2.1 网络设计

我们将该模型实现为卷积神经网络，并在PASCAL VOC检测数据集上对其进行评估[9]。网络的初始卷积层从图像中提取特征，而全连接层则预测输出概率和坐标。

我们的网络体系结构受到用于图像分类的GoogLeNet模型的启发[34]。 我们的网络有24个卷积层，其后是2个全连接层。与Lin等[22]相似，我们没有使用GoogLeNet的初始模块，而是简单地使用1×1 还原层（reduction layers）和3×3卷积层。 完整的网络如图3所示。

我们还训练了一种快速版本的YOLO，旨在突破快速目标检测的界限。Fast YOLO使用的神经网络具有较少的卷积层（9个而不是24个），并且这些层中的过滤器较少。 除了网络的规模外，YOLO和Fast YOLO之间的所有训练和测试参数都相同。

![image-20200408230958018](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200408230958018.png)

我们的网络的最终输出是预测的7×7×30的张量。

## 2.2 训练

我们在ImageNet 1000类竞赛数据集上预训练卷积层[30]。对于预训练，我们使用图3中的前20个卷积层，然后是平均池化层和全连接层。我们对该网络进行了大约一周的训练，并在ImageNet 2012验证集上实现了88%的single crop top-5准确率，与Caffe’s Model Zoo [24].的GoogLeNet模型相当。我们使用Darknet框架进行所有训练和推理[26]。

然后我们将模型转换为执行检测。Ren等人表明将卷积层和连接层添加到预训练网络可以提高性能[29]。以它们为例，我们添加了四个卷积层和两个具有随机初始化权重的全连接层。检测通常需要细粒度的视觉信息，因此我们将网络的输入分辨率从224 × 224提高到448 × 448。

最后一层预测类概率和边界框坐标。我们通过图像的宽度和高度来规范化边界框的宽度和高度，使它们介于0和1之间。我们将边界框的x和y坐标参数化为特定网格单元位置的偏移，因此它们也被限定在0和1之间。

我们对最终层使用线性激活函数，所有其他层使用以下有漏隙的校正线性激活函数（leaky rectified linear activation）：

![image-20200409154350282](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200409154350282.png)

我们对模型输出的平方和误差进行了优化。我们使用平方和误差，因为它很容易优化，但它并不完全符合我们最大化平均精度的目标。它将定位误差与分类误差平均加权，分类误差可能不理想。此外，在每个图像中，许多网格单元不包含任何物体。这会将这些单元格的“置信度”分数推向零，通常会压倒（overpowering）包含目标的单元格的梯度。这可能会导致模型不稳定，导致训练提前发散。

为了弥补这一点，我们增加了边界框坐标预测的损失，并减少了不包含目标的框的置信预测的损失。我们使用两个参数，λ_coord和λ_noobj来实现这一点。我们将λ_coord设为5，λ_noobj设为.5。

平方和误差在大boxes和小boxes中的权重相等。我们的误差度量应该反映出大boxes里的小偏差比小boxes里的小偏差更重要。为了部分解决这个问题，我们预测边界框宽度和高度的平方根，而不是直接预测宽度和高度。

YOLO在每个网格单元有多个预测边界框。在训练时，我们只希望一个边界框预测器负责每个目标。我们指定一个预测器负责预测一个目标，这个目标是基于预测框和真实值有最高的IOU。这将导致边界框预测器之间的特殊化。每个预测器在预测特定大小、宽高比或物体类别方面都会做得更好，从而提高整体recall能力。

在训练期间，我们优化了以下多部分损失函数：

![image-20200409165535466](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200409165535466.png)

其中，![image-20200409165639190](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200409165639190.png)表示目标是否出现在单元格i中，![image-20200409165714596](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200409165714596.png)表示单元格i中的第j个边界框预测器对该预测负责。

注意，如果网格单元中存在物体，则loss函数只惩罚分类错误（前面讨论的条件类概率）。如果预测器是负责的真实框（即，在这个网格单元中任何和真实值有最高的IOU的预测框），它也只惩罚边界框坐标错误。

我们在PASCAL VOC 2007和2012的训练和验证数据集上对网络进行了大约135个epochs的训练。在2012数据集测试时，我们还包括了用于训练的VOC 2007测试数据。在整个训练过程中，我们使用64个批次，动量为0.9，衰减为0.0005。

我们的学习率变化如下：在第一个阶段，我们将学习率从10^-3 缓慢提高到10^ -2。
如果我们从一个高学习率开始，我们的模型经常因为不稳定的梯度而发散。我们开始以10^-2 训练75个epoch，接着10^-3 训练30个epoch，最后10^-4 训练30个epoch。

为了避免过拟合，我们使用了droupout和大量的数据增强。在第一个连接层之后添加了rate为0.5的droupout以防止层之间的协同适应（co-adaptation）[18]。对于数据增强，我们引入了最大高于原始图像大小20%的随机缩放和平移。在factor为1.5的HSV颜色空间中，我们还随机调整图像的曝光度和饱和度。

## 2.3 推理

就像在训练中一样，预测测试图像的检测只需要一个网络评估。在PASCAL VOC上，网络预测每个图像有98个边界框，每个box有类概率。与基于分类器的方法不同，YOLO在测试时速度非常快，因为它只需要一个网络评估。

网格设计在边界框预测中增强了空间多样性。通常很清楚目标属于哪个网格单元，并且网络只为每个目标预测一个框。但是，一些大型目标或靠近多个单元格边界的目标可以被多个单元格很好地定位。非最大值抑制可以用来修正这些多重检测。对于R-CNN或DPM来说，非最大值抑制对性能并不重要，虽然mAP增加了2-3%。

## 2.4 YOLO的不足

YOLO对边界框预测施加了很强的空间约束，因为每个网格单元只能预测两个框，并且只能有一个类。这个空间约束限制了我们的模型可以预测的邻近目标的数量。我们的模型对于成群出现的小物体很挣扎，例如成群的鸟。

由于我们的模型学习从数据中预测边界框，所以它很难推广到新的或不寻常的宽高比或配置的物体。我们的模型还使用相对粗糙的特征来预测边界框，因为我们的架构有来自输入图像的多个下采样层。

最后，当我们训练一个接近检测性能的损失函数时，我们的损失函数对待小边界框和大边界框中的错误是一样的。大box里的小错误通常是良性的，但小box里的小错误对IOU的影响要大得多。我们的主要错误来源是定位的错误。

# 3. 与其他检测系统的比较

目标检测是计算机视觉的核心问题。检测pipeline通常首先从输入图像中提取一组鲁棒特征（Haar[25]、SIFT[23]、HOG[4]、卷积特征[6]）。然后，使用分类器[36、21、13、10]或定位器[1、32]来识别特征空间中的物体。这些分类器或定位器要么以滑动窗口的方式在整个图像上运行，要么在图像中的某些区域子集上运行[35、15、39]。
我们将YOLO检测系统与几个顶级的检测框架进行了比较，突出了关键的相似性和差异性。

**DPM。**可变形零件模型（DPM）使用滑动窗口方法进行目标检测[10]。DPM使用一个不相交的pipeline来提取静态特征、分类区域、预测高分区域的边界框等，我们的系统用一个卷积神经网络来代替所有这些不同的部分。该网络同时执行特征提取、边界盒预测、非最大值抑制和上下文推理。网络不是静态特征，而是在线训练特征，并针对检测任务对其进行优化。与DPM模型相比，我们的统一架构带来了更快、更精确的模型。

**R-CNN。**R-CNN及其变体使用区域建议而不是滑动窗口来查找图像中的目标。选择性搜索[35]生成潜在的边界框，卷积网络提取特征，支持向量机对box打分，线性模型调整边界框，非最大值抑制消除重复检测。这个复杂pipeline的每个阶段都必须独立进行精确的调整，结果系统非常慢，在测试时每张图像需要40秒以上。

YOLO和R-CNN有一些相似之处。每个网格单元提出潜在的边界框，并使用卷积特征对这些边界盒进行评分。然而，我们的系统在网格单元建议上设置空间约束，这有助于减少对同一目标的多次检测。我们的系统也提出了更少的边界框，每张图只有98个，而选择性搜索有将近2000个。最后，我们的系统将这些单独的组件组合成一个单独的、联合优化的模型。

**其他快速检测器。**Fast和Faster R-CNN关注于通过共享计算和使用神经网络提出区域而不是选择性搜索来加速R-CNN框架[14] [28]。虽然它们比R-CNN提供了速度和准确性的改进，但都还不能达到实时性能。

许多研究工作集中在加速DPM pipeline[31] [38] [5]。它们加速HOG计算，使用级联，并将计算推送到gpu。然而，只有30hz DPM[31]能够实时运行。

YOLO没有试图优化大型检测pipeline的单个组件，而是完全抛弃pipeline，并且设计得很快。

像人脸或人这样的单个类的检测器可以高度优化，因为它们必须处理更少的变化[37]。YOLO是一种通用的探测器，它可以学习同时检测各种物体。

**Deep MultiBox。**与R-CNN不同，Szegedy等人，训练卷积神经网络来预测感兴趣的区域[8]，而不是使用选择性搜索。MultiBox还可以通过将置信度预测替换为单类预测来执行单目标检测。然而，MultiBox不能进行一般的目标检测，仍然只是一个较大的检测pipeline中的一部分，需要进一步的补充图像分类。YOLO和MultiBox都使用卷积网络来预测图像中的边界框，但YOLO是一个完整的检测系统。

**OverFeat。**Sermanet等人，训练卷积神经网络来执行定位，并使该定位器调整执行检测[32]。OverFeat有效地执行滑动窗口检测，但它仍然是一个不相交的（disjoint）系统。OverFeat优化了定位，而不是检测性能。像DPM一样，定位程序在进行预测时只看到局部信息。OverFeat不能解释全局上下文，因此需要大量的后处理来产生一致的检测。

**MultiGrasp。**我们的工作在设计上与Redmon等人的抓取检测工作类似[27]。我们的边界框预测的网格方法是基于多抓取系统的回归抓取。然而，抓取检测比目标检测简单得多。对于包含一个目标的图像，多重抓取只需要预测一个可抓取区域。它不需要估计物体的大小、位置或边界，也不需要预测其类别，只需要找到一个适合抓取的区域。YOLO预测图像中多个类的多个目标的边界框和类概率。

# 4. 实验

首先我们比较了YOLO和其他实时检测系统在PASCAL VOC 2007上的性能。为了了解YOLO和R-CNN变体之间的差异，我们探讨了YOLO和Fast R-CNN在VOC 2007上的错误，后者是R-CNN性能最高的版本之一[14]。基于不同的误差分布，我们证明YOLO可以用来重新打分（rescore）Fast R-CNN的检测，减少背景误报带来的误差，从而显著提高性能。我们还展示了VOC 2012的结果，并将mAP与当前最先进的方法进行了比较。最后，我们证明YOLO在两个图形数据集上比其他检测器更好地推广到新域。

## 4.1 与其他实时系统的对比

目标检测的许多研究工作都集中在快速建立标准检测pipeline上。[5]  [38] [31] [14] [17] [28]然而，只有Sadeghi等人。实际产生了一个实时运行的检测系统（每秒30帧或更好）[31]。我们将YOLO与他们的DPM的GPU实现进行了比较，DPM的运行频率可以是30Hz，也可以是100Hz。虽然其他的efforts没有达到实时里程碑，我们也比较了他们的相对mAP和速度，以检查目标检测系统的准确性与性能折衷的可能性。

Fast YOLO是PASCAL上最快的目标检测方法；据我们所知，它是现存最快的目标检测方法。在52.7%的mAP下，实时检测的准确率是以往工作的两倍以上。YOLO将mAP推到63.4%，同时仍然保持实时性能。

我们还使用VGG-16训练YOLO。该模型比YOLO模型更精确，但速度明显慢。它有助于与依赖VGG-16的其他检测系统进行比较，但由于它比实时性慢，本文的其余部分将重点放在我们更快的模型上。

最快的DPM有效地加快了DPM的速度，而不会牺牲太多的mAP，但它仍然没有factor为2时的实时性能[38]。与神经网络方法相比，DPM的检测精度相对较低。

R-CNN用静态边界框建议代替选择性搜索[20]。虽然它比R-CNN快的多，但是它仍然缺乏实时性，并且由于没有好的proposal而受到了很大的准确性打击。

Fast R-CNN加速了R-CNN的分类阶段，但它仍然依赖于选择性搜索，每幅图像大约需要2秒来生成边界框proposal。因此，它有很高的mAP，但速度是0.5fps，所以它仍然和实时差的很远。

最近Faster R-CNN用一个神经网络代替了选择性搜索来提出边界框，类似于Szegedy等人。[8] 在我们的测试中，他们最精确的模型达到了每秒7帧，而较小的，不太精确的模型达到了每秒18帧。VGG-16版本的Faster R-CNN比YOLO高10 mAP，但也比YOLO慢6倍。ZeilerFergus的R-CNN速度比YOLO慢2.5倍，而且也不太准确。

![image-20200409180106019](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200409180106019.png)

## 4.2 VOC2007 误差分析

为了进一步研究YOLO和最先进探测器之间的差异，我们查看了VOC 2007的详细结果分类。我们将YOLO与Fast R CNN进行比较，因为Fast R-CNN是PASCAL上性能最高的检测器之一，而且它的检测是公开的。

我们使用了Hoiem等人的方法和工具。[19] 对于测试时的每个类别，我们查看该类别的前N个预测。每个预测要么正确，要么根据错误类型进行分类：

![image-20200409180121364](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200409180121364.png)

<img src="C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200409180131291.png" alt="image-20200409180131291"  />

图4显示了所有20个类中平均每个错误类型的细分。

![image-20200409180220466](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200409180220466.png)

YOLO比较难正确地定位目标。定位错误比所有其他来源加起来都要多。Fast R-CNN定位误差小得多，但背景误差大得多。误差最多的13.6%的检测是不包含任何目标的误报。Fast R-CNN比YOLO更容易预测背景检测。

## 4.3 对比YOLO和Fast R-CNN

YOLO的背景错误比Fast R-CNN少得多。通过使用YOLO消除Fast R-CNN的背景检测，我们在性能上得到了显著的提高。对于R-CNN预测的每个边界框，我们检查YOLO是否预测了类似的框。如果是的话，我们会根据YOLO预测的概率和两个框之间的重叠来提高预测。

最佳的Fast R-CNN模型在VOC 2007测试集上实现了71.8%的mAP。当与YOLO结合时，mAP增加3.2%至75.0%。我们还尝试将最佳的Fast R-CNN模型与其他几个版本的Fast R-CNN相结合。这些组合使mAP获得了小幅度的在0.3%到0.6%之间的增长，详情见表2。

![image-20200409180739108](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200409180739108.png)

来自YOLO的提升不仅仅是模型集成的副产品，因为结合不同版本的Fast R-CNN几乎没有什么好处。相反，正是因为YOLO在测试时犯了各种各样的错误，所以它对提高Fast R-CNN的性能非常有效。

不幸的是，这种组合并没有受益于YOLO的速度，因为我们分别运行每个模型，然后将结果组合起来。然而，由于YOLO是如此之快，相比Fast R-CNN，它没有增加任何显著的计算时间。

## 4.4 VOC 2012 的结果

在VOC 2012测试集中，YOLO的mAP分数为57.9%。这比目前的技术水平低，更接近使用VGG-16的原始R-CNN，见表3。

![image-20200409181250217](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200409181250217.png)

与最接近的竞争对手相比，我们的系统在处理小物体方面存在困难。在奶瓶、绵羊和电视/监视器等类别中，YOLO的得分比R-CNN或Feature Edit低8-10%。但是，在其他类别上，如cat和train， YOLO可以获得更高的性能。

我们的组合Fast R-CNN+YOLO模型是性能最好的检测方法之一。Fast R-CNN与YOLO的结合提高了2.3%，在公共排行榜上提升了5个位置。

## 4.5 泛化性能：艺术品中的人的检测

用于目标检测的学术数据集从相同的分布中提取训练和测试数据。在实际应用中，很难预测所有可能的用例，测试数据可能与系统之前看到的不同[3]。我们将YOLO与毕加索数据集[12]和People Art数据集[3]上的其他检测系统进行了比较，这两个数据集用于测试艺术品上的人的检测。

![image-20200409181650186](C:\Users\WangIronman\AppData\Roaming\Typora\typora-user-images\image-20200409181650186.png)

图5显示了YOLO和其他检测方法的比较性能。作为参考，我们给出了VOC 2007的人身检测AP，所有的模型都是在voc2007数据上训练的。毕加索的模型接受的是VOC 2012的训练，而在人物艺术方面，他们接受的是VOC 2010的训练。

R-CNN在VOC 2007上有很高的AP。然而，R-CNN在应用于艺术作品时精度会大幅度下降。R-CNN使用选择性搜索的边界框建议，这是调整为自然图像。R-CNN中的分类步骤只看到小区域，需要好的proposal。

DPM在应用于艺术品时能很好地保持它的AP。先前的工作理论认为，DPM表现良好，因为它有强大的空间模型的形状和目标的布局。尽管DPM没有R-CNN下降那么多，但它从一个较低的AP开始。

YOLO在VOC 2007上有很好的性能，其AP在应用于艺术品时的下降的低于其他方法。与DPM一样，YOLO对目标的大小和形状、目标之间的关系以及目标通常出现的位置进行建模。艺术品和自然图像在像素级别上有很大的不同，但它们在物体的大小和形状上是相似的，因此YOLO仍然可以预测良好的边界框和检测。

# 5. 世界上的实时检测

YOLO是一种快速、准确的目标检测器，非常适合计算机视觉应用。我们将YOLO连接到网络摄像头并验证它是否保持实时性能，包括从摄像机获取图像和显示检测结果的时间。

由此产生的系统是互动的和引人入胜的。当YOLO单独处理图像时，当连接到网络摄像头时，它的功能就像一个跟踪系统，在目标移动和外观变化时检测它们。系统演示和源代码可以在我们的项目网站上找到：http://pjreddie.com/yolo/。

# 6. 总结

本文介绍了一种统一的目标检测模型YOLO。我们的模型构造简单，可以直接在全图像上训练。与基于分类器的方法不同，YOLO是基于与检测性能直接对应的损失函数来训练的，整个模型是联合训练的。Fast YOLO是目前文献中速度最快的通用目标检测器，它推动了实时目标检测技术的发展。YOLO还可以很好地推广到新的领域，使其成为依赖于快速、鲁棒的目标检测的应用程序的理想选择。

# 参考文献

[1] M. B. Blaschko and C. H. Lampert. Learning to localize objects
with structured output regression. In Computer Vision–
ECCV 2008, pages 2–15. Springer, 2008. 4
[2] L. Bourdev and J. Malik. Poselets: Body part detectors
trained using 3d human pose annotations. In International
Conference on Computer Vision (ICCV), 2009. 8
[3] H. Cai, Q. Wu, T. Corradi, and P. Hall. The crossdepiction
problem: Computer vision algorithms for recognising
objects in artwork and in photographs. arXiv preprint
arXiv:1505.00110, 2015. 7
[4] N. Dalal and B. Triggs. Histograms of oriented gradients for
human detection. In Computer Vision and Pattern Recognition,

CVPR 2005. IEEE Computer Society Conference
on, volume 1, pages 886–893. IEEE, 2005. 4, 8
[5] T. Dean, M. Ruzon, M. Segal, J. Shlens, S. Vijayanarasimhan,
J. Yagnik, et al. Fast, accurate detection of
100,000 object classes on a single machine. In Computer
Vision and Pattern Recognition (CVPR), 2013 IEEE Conference
on, pages 1814–1821. IEEE, 2013. 5
[6] J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang,
E. Tzeng, and T. Darrell. Decaf: A deep convolutional activation
feature for generic visual recognition. arXiv preprint
arXiv:1310.1531, 2013. 4
[7] J. Dong, Q. Chen, S. Yan, and A. Yuille. Towards unified
object detection and semantic segmentation. In Computer
Vision–ECCV 2014, pages 299–314. Springer, 2014. 7
[8] D. Erhan, C. Szegedy, A. Toshev, and D. Anguelov. Scalable
object detection using deep neural networks. In Computer
Vision and Pattern Recognition (CVPR), 2014 IEEE Conference
on, pages 2155–2162. IEEE, 2014. 5, 6
[9] M. Everingham, S. M. A. Eslami, L. Van Gool, C. K. I.
Williams, J. Winn, and A. Zisserman. The pascal visual object
classes challenge: A retrospective. International Journal
of Computer Vision, 111(1):98–136, Jan. 2015. 2
[10] P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ramanan.
Object detection with discriminatively trained part
based models. IEEE Transactions on Pattern Analysis and
Machine Intelligence, 32(9):1627–1645, 2010. 1, 4
[11] S. Gidaris and N. Komodakis. Object detection via a multiregion
& semantic segmentation-aware CNN model. CoRR,
abs/1505.01749, 2015. 7
[12] S. Ginosar, D. Haas, T. Brown, and J. Malik. Detecting people
in cubist art. In Computer Vision-ECCV 2014Workshops,
pages 101–116. Springer, 2014. 7
[13] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature
hierarchies for accurate object detection and semantic
segmentation. In Computer Vision and Pattern Recognition
(CVPR), 2014 IEEE Conference on, pages 580–587. IEEE,1.1, 4, 7
[14] R. B. Girshick. Fast R-CNN. CoRR, abs/1504.08083, 2015.
2, 5, 6, 7
[15] S. Gould, T. Gao, and D. Koller. Region-based segmentation
and object detection. In Advances in neural information
processing systems, pages 655–663, 2009. 4
[16] B. Hariharan, P. Arbel´aez, R. Girshick, and J. Malik. Simultaneous
detection and segmentation. In Computer Vision–
ECCV 2014, pages 297–312. Springer, 2014. 7
[17] K. He, X. Zhang, S. Ren, and J. Sun. Spatial pyramid pooling
in deep convolutional networks for visual recognition. arXiv
preprint arXiv:1406.4729, 2014. 5
[18] G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and
R. R. Salakhutdinov. Improving neural networks by preventing
co-adaptation of feature detectors. arXiv preprint
arXiv:1207.0580, 2012. 4
[19] D. Hoiem, Y. Chodpathumwan, and Q. Dai. Diagnosing error
in object detectors. In Computer Vision–ECCV 2012, pages
340–353. Springer, 2012. 6
[20] K. Lenc and A. Vedaldi. R-cnn minus r. arXiv preprint
arXiv:1506.06981, 2015. 5, 6
[21] R. Lienhart and J. Maydt. An extended set of haar-like features
for rapid object detection. In Image Processing. 2002.
Proceedings. 2002 International Conference on, volume 1,
pages I–900. IEEE, 2002. 4
[22] M. Lin, Q. Chen, and S. Yan. Network in network. CoRR,
abs/1312.4400, 2013. 2
[23] D. G. Lowe. Object recognition from local scale-invariant
features. In Computer vision, 1999. The proceedings of the
seventh IEEE international conference on, volume 2, pages
1150–1157. Ieee, 1999. 4
[24] D. Mishkin. Models accuracy on imagenet 2012
val. https://github.com/BVLC/caffe/wiki/
Models-accuracy-on-ImageNet-2012-val. Accessed:
2015-10-2. 3
[25] C. P. Papageorgiou, M. Oren, and T. Poggio. A general
framework for object detection. In Computer vision, 1998.
sixth international conference on, pages 555–562. IEEE,1.4
[26] J. Redmon. Darknet: Open source neural networks in c.
http://pjreddie.com/darknet/, 2013–2016. 3
[27] J. Redmon and A. Angelova. Real-time grasp detection using
convolutional neural networks. CoRR, abs/1412.3128, 2014.5
[28] S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: Towards
real-time object detection with region proposal networks.
arXiv preprint arXiv:1506.01497, 2015. 5, 6, 7
[29] S. Ren, K. He, R. B. Girshick, X. Zhang, and J. Sun. Object
detection networks on convolutional feature maps. CoRR,
abs/1504.06066, 2015. 3, 7
[30] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh,
S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein,
A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual
Recognition Challenge. International Journal of Computer
Vision (IJCV), 2015. 3
[31] M. A. Sadeghi and D. Forsyth. 30hz object detection with
dpm v5. In Computer Vision–ECCV 2014, pages 65–79.
Springer, 2014. 5, 6
[32] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus,
and Y. LeCun. Overfeat: Integrated recognition, localization
and detection using convolutional networks. CoRR,
abs/1312.6229, 2013. 4, 5
[33] Z. Shen and X. Xue. Do more dropouts in pool5 feature maps
for better object detection. arXiv preprint arXiv:1409.6911,1.7
[34] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed,
D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich.
Going deeper with convolutions. CoRR, abs/1409.4842,1.2
[35] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W.
Smeulders. Selective search for object recognition. International
journal of computer vision, 104(2):154–171, 2013.4
[36] P. Viola and M. Jones. Robust real-time object detection.
International Journal of Computer Vision, 4:34–47, 2001. 4
[37] P. Viola and M. J. Jones. Robust real-time face detection.
International journal of computer vision, 57(2):137–154,1.5
[38] J. Yan, Z. Lei, L. Wen, and S. Z. Li. The fastest deformable
part model for object detection. In Computer Vision and Pattern
Recognition (CVPR), 2014 IEEE Conference on, pages
2497–2504. IEEE, 2014. 5, 6
[39] C. L. Zitnick and P. Doll´ar. Edge boxes: Locating object proposals
from edges. In Computer Vision–ECCV 2014, pages
391–405. Springer, 2014. 4