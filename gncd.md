mask是一连串是否为新类的01标志位，旧类为true，新类为false/有标记数据或无标记数据

mask.mean()计算标记与无标记比例

project_utils

> cluster_utils.py计算聚类精确度组件，线性任务组件
>
> 

bash_scripts

> constrastive_train.sh微调训练
>
> extract_features.sh微调训练结束后，直接用于测试，提取特征向量
>
> estimate_k.sh测试过程中的k估计
>
> k_means.sh测试过程中的基于已有特征向量的kmeans聚类