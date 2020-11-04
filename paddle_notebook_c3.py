#paddle

#三种模型资源
"""
预训练模型工具 paddleHub：python API 一行代码完成预训练模型的预测，Fine-tune API 10行代码完成迁移学习，进行原型验证
特定场景开发套件 paddleX：paddleClas,paddleDetection,paddleSeg
开源模型库 paddle Models: 开源模型代码，直接运行或修改

paddleHub->paddleX->paddle Models
"""

#工业部署
"""
paddle inference 服务器端模型部署
paddle serving 云端服务化部署
paddle lite 轻量化推理引擎，用于Mobile和lOT场景
"""

#模型压缩工具
"""
paddle slim 获得更小体积的模型和更快的执行速度
"""

#模型转换
"""
X2paddle 将其他框架模型转换为paddle模型
"""

#辅助工具
"""
AutoDL 自动化深度学习工具，自动搜索最优的网络结构与超参数，免去用户在诸多网络结构中选择困难和人工调参
VisualDL 可视化分析工具
"""

#paddleHub
"""
预训练模型应用工具
python API 调用模型
Fine-tune API 内置多种优化策略，完成预训练模型的fine-tune
模型转服务
自动超参数学习，内置AutoDL finetuner自动化超参搜索
"""
"""
hub install/uninstall 模型的升级和卸载
"""

#paddle models
"""
深度学习算法集合，包括代码、数据集、预训练模型
paddle models的文档提供了两层索引：
第一层索引以展示任务输入和输出的形式，供用户确定他的任务属于哪一类问题；
第二层索引详细列出了在该类问题中存在的诸多模型究竟有是很么区别，通常用户可以从模型精度、训练和预测速度、模型体积和适用于特定场景来决定使用哪个模型，对于历史上的经典模型以从学习的视角提供了实现
"""