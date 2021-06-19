# Huawei-2019-AI-Competition
“华为云杯”2019人工智能创新应用大赛 Top10%

比赛链接 ：
[“华为云杯”2019人工智能创新应用大赛](https://competition.huaweicloud.com/information/1000021526/circumstances?track=107)

# 赛题任务
&ensp;初赛赛题任务是对西安的热门景点、美食、特产、民俗、工艺品等图片进行分类，即首先识别出图片中物品的类别（比如大雁塔、肉夹馍等），然后根据图片分类的规则，输出该图片中物品属于景点、美食、特产、民俗和工艺品中的哪一种。共有54个类别。

# 输入输出格式
<b>输入数据格式：</b>   
<img src="https://github.com/ielym/Huawei-2019-AI-Competition/blob/main/imgs/img_12.jpg" width="30%">     
<b>输出数据格式：</b>      
{   
  " result ": "工艺品/仿唐三彩"   
}

# 算法模型
***    
刚好研究生入学确定方向后不久就碰到了这个比赛，比赛大概一个多月时间，按照 <b>Baseline（ResNet50）</b> 和大佬分享经验开始接触图像分类经典模型论文和代码，包括：     
* <b>AlexNet</b> ： 害！ 本科毕设用的就是这种结构，深度学习好简单！      
* <b>VGGNets</b> ： 网络越深越复杂越好啊！     
* <b>ZFNet</b> ：原来CNN是用来提取特征的！原来网络各层提取的特征是有分工的！原来深层网络能够融合浅层边缘纹理颜色特征组合出更高维的高级语义特征！原来VGG效果好也是因为这个原因啊！     
* <b>ResNet</b> ：何恺明yyds！直接被惊艳到的一篇，原来CNN不只是Plain Network这一种结构，对何恺明大神的关注也是从这里开始！      
* <b>SENet</b> ：开始知道了有"Attention"这种魔法，效果也是出奇的好啊！      
* <b>GoogleNets</b> ： 原来卷积还能并行着连接！！并行串行跳跃连接的结构都看过了，是不是可以随心所欲搭积木了！     
* <b>直到...看到比赛群里大佬分享的EfficientNets       </b>
这是什么神仙操作？？？原来深度学习模型效果和模型深度/宽度，还和输入图像分辨率都有关啊！而且好像还不是模型越复杂越好，另外有钱确实任性。

所以，这次比赛最终使用<b>Efficient-b6</b>模型，借助比赛提供的免费Model Arts平台进行了模型训练（本地1080ti实在是训练不动）     
最终，获得了线上95.9的准确率评分，最终排名是top 10%。

# 数据处理
虽然本科毕设方向是图像分类，但在参加这个比赛之前，心中坚定的认为模型为王，从来没有认真体会过训练数据对算法性能的影响。     
比赛刚开始，不知道训练集类别分布平不平衡，训练集/测试集图像边缘分布一不一致，标签分布一不一致，以及会有什么影响，只能按照群里大佬分享一点方案一点一点扩展知识点并尝试。     
逐渐的，对数据在深度学习中的重要性，特别是数据分布的重要性的认识也越来越清晰（包括后续打比赛，首先都要详细的进行数据分析，厚积薄发才是最重要的）。     
在阅读经典分类模型的过程中，一直围绕着 <b> Imagenet </b> 这个数据集， 不过之前认为，不就是个数据集嘛，我要看模型网络结构设计，学习怎么搭积木！ 不过现在发现，数据对于评分的提高是更cheap的方案！
因此比赛中首先是对图像进行了水平垂直翻转，随机角度旋转：     
<img src="https://github.com/ielym/Huawei-2019-AI-Competition/blob/main/imgs/img_12_3.jpg" width="30%">

此外，使用爬虫按照类别关键字爬取了百度图片近一万张额外图像数据，但.... 其中有很多噪声图像，怎么打标签？？？说实话，前期确实通宵达旦手工打标签，直到
看到了<b>伪标签法</b>这种奇技淫巧，了解了之后更是发现原来除了监督学习之外还有半监督学习！

# 超参数调整
这次比赛对于超参数的优化还是一头雾水的，一想到可能调整一个超参数，其他超参数都要重新调整就头疼！ 况且也没有足够的资源去做更多更快的尝试，通常更改一次初始学习率的值都要花费大半天时间进行模型训练。不过偶然还是发现了在迁移学习时，学习率调小到1e-4是比1e-3更好的。
也基于这个发现，开始了各种学习率衰减策略的尝试。

# 比赛总结
回过头来看，这次比赛可以看作是我CV入门的启蒙导师。 回顾比赛过程，简单做个总结：
* 由于深度学习的可解释性较弱，因此理论+实践的方式绝对算得上是入门过程中绝佳的学习路线了。比赛时虽然还不懂得什么高级的可视化技术，但是对于改动什么地方能够对模型有什么影响的这种感觉已经渐渐形成了，也更方便了后续的深入学习。
* 比赛中，从深度学习简单模块的编写，到简单模型整体的复现，对我后来代码能力的提升是极其重要的。
* 当时同时做着尝试各种网络结构，数据增强方法和超参数调整，沉下心来做了一个多月之后，对于后来对深度学习的理解也是十分有帮助的。
* 第一次参加CV比赛，完成了从小白到青铜的过度阶段，并且获得了Top 10%的成绩。再加上能够通过可视化的方式对输入图像进行直观的输入输出，对于我对CV领域的兴趣培养也是詪有帮助的，从此科研不再枯燥！卷入CV无悔~

***
足够幸运，才能在研究生入学不久就遇见这场图像分类比赛带我走进CV的世界     


<img src="http://tiebapic.baidu.com/forum/pic/item/488e8495a4c27d1e504795850cd5ad6edcc43844.jpg" width="50%">
***



