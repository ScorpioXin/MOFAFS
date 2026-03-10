MOFAFS（Multi-Objective Feeding Allocation and Feeding Scheduling）
多目标投喂调度优化框架

本仓库提供 MOFAFS 多目标投喂调度优化算法的实现代码。该方法面向复杂资源约束与动态环境下的投喂调度问题，通过多目标优化算法实现资源分配与调度策略的协同优化，提高系统运行效率与资源利用率。

该框架适用于以下研究场景：
工厂化水产养殖投喂调度
多目标调度优化问题
资源分配与智能决策
调度算法实验与性能评估

环境配置
建议使用 conda 环境。
1 克隆仓库
git clone https://github.com/ScorpioXin/MOFAFS.git
cd MOFAFS
2 创建环境
conda create -n mofafs python=3.9
conda activate mofafs

MOFAFS 框架面向 多目标投喂调度优化问题，综合考虑：
养殖生物生长需求
投喂资源约束
调度效率
系统运行成本
通过多目标优化算法搜索最优投喂策略，在不同目标之间实现平衡，从而获得高效稳定的调度方案。

实验结果保存在：
results/
目录中。

如果本项目对您的研究有所帮助，请引用：
@misc{MOFAFS,
  author = {Xin},
  title = {MOFAFS: 多目标投喂调度优化框架},
  year = {2026},
  url = {https://github.com/ScorpioXin/MOFAFS}
}
