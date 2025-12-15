# Jiaotong_Guihua_Yuanli
项目简介
本软件是一个基于Python开发的交通分配计算工具，实现了交通规划中常用的多种交通流分配算法。软件能够读取路网结构和出行需求数据，进行交通分配计算，并生成可视化结果和详细分析报告。

主要功能
网络数据加载：支持从JSON格式文件加载道路网络和出行需求数据

多种分配算法：
  全有全无分配 (All-or-Nothing,AON)
  增量分配 (Incremental Assignment)
  基于 Frank-Wolfe 算法的用户均衡分配 (User Equilibrium)
  路径分析：Dijkstra 算法寻找最短路径，支持考虑/不考虑拥堵效应

结果可视化：
  网络结构图
  流量分配图
  算法对比图
  收敛曲线图
定量评估：计算路网总出行时间，评估分配效率

模块文件	                              功能描述
  main.py	                      主程序，协调各模块执行
  traffic_network.py	        交通网络数据结构和基本操作
  assignment_algorithms.py	        交通分配算法实现
  visualization.py	                结果可视化模块

数据文件
  文件	    	                  内容
  network.json		        路网节点和路段信息
  demand.json		           出行需求（OD矩阵）
  
环境要求
  Python 3.7+

依赖库：
  numpy
  matplotlib
  scipy

安装步骤
  克隆或下载项目代码

安装依赖库：
  pip install numpy matplotlib scipy

准备数据文件：
  network.json：路网结构数据
  demand.json：出行需求数据

运行软件
  python main.py

运行流程
  加载网络和需求数据
  执行多种分配算法
  生成可视化图表
  输出详细分析报告

功能详解
1. 网络加载与解析
  软件能够正确解析 PDF 中描述的测试场景，包括：
  7个节点 (A,B,C,D,E,F,G)
  8个双向路段
  6个OD对，总需求6000辆/小时

2. 分配算法实现
  全有全无分配 (AON)
    基于自由流时间的最短路径分配，不考虑拥堵效应，作为基准分配方法
  增量分配
    将需求分成多个增量，每轮分配后更新行程时间，逐步逼近均衡状态 
  用户均衡分配 (UE)
    基于 Frank-Wolfe 算法，实现 Wardrop 第一原理，保证所有被使用路径行程时间相等

3. 问题回答功能
  软件能够自动回答课程要求的四个问题：
    不考虑拥堵的最快路径：显示A到F的最短路径及行程时间
    考虑拥堵的最快路径：在拥堵效应下的最快路径
    单 OD 对分析：分析A到F需求的分配结果
    全网络分析：比较不同算法的总出行时间和效率
  输出结果
    软件运行后会在output/目录生成以下文件：
    network_structure.png：网络结构示意图   
    aon_assignment.png：AON分配结果 
    incremental_assignment.png：增量分配结果
    ue_assignment.png：用户均衡分配结果 
    flow_comparison.png：流量对比图
    convergence_curve.png：UE算法收敛曲线
    results_summary.txt：详细结果汇总

关键算法代码说明
  Dijkstra最短路径算法
    见 assignment_algorithms.py中的dijkstra_shortest_path方法，支持考虑/不考虑拥堵的两种模式
  
  Frank-Wolfe 用户均衡算法
    见 assignment_algorithms.py中的frank_wolfe_ue方法，实现Wardrop均衡原理，使用黄金分割法寻找最优步长
  BPR 行程时间函数
    行程时间计算基于标准 BPR 函数：t(q) = t₀ * (1 + q/cap)²，其中 t₀ 为自由流时间，q 为流量，cap 为通行能力

测试场景分析
  网络特征
    路段容量：1800 或 3600 辆/小时
    最大限速：30 或 60 公里/小时
    双向交通，总需求 6000 辆/小时
  
  分配结果特点
    AON 分配：流量高度集中，部分路段过载
    增量分配：流量分布更加合理，总时间减少
    UE分配：达到均衡状态，总时间最优

使用示例
  基本使用
    from traffic_network import TrafficNetwork
    from assignment_algorithms import AssignmentAlgorithms
    
    # 加载网络
    network = TrafficNetwork("network.json")
    
    # 进行AON分配
    aon_flow = AssignmentAlgorithms.all_or_nothing(network, demand)
    
    # 计算总出行时间
    total_time = AssignmentAlgorithms.calculate_total_travel_time(network, aon_flow)
    单OD对分析
    python
    # 分析A到F的分配
    link_flow, path_details = AssignmentAlgorithms.analyze_single_od_assignment(
        network, "A", "F", 2000, "UE"
    )
