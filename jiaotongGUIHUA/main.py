#!/usr/bin/env python3
"""
交通分配计算软件 - 主程序
作者: [你的姓名]
学号: [你的学号]
"""

import json
import os
from typing import Dict, Tuple
import numpy as np

# 导入自定义模块
from traffic_network import TrafficNetwork
from assignment_algorithms import AssignmentAlgorithms
from visualization import TrafficVisualizer

def load_demand(demand_file: str) -> Dict[Tuple[str, str], float]:
    """
    加载需求数据
    
    Args:
        demand_file: 需求JSON文件路径
        
    Returns:
        OD需求字典
    """
    with open(demand_file, 'r') as f:
        data = json.load(f)
    
    demand = {}
    for i in range(len(data['from'])):
        origin = data['from'][i]
        destination = data['to'][i]
        amount = data['amount'][i]
        demand[(origin, destination)] = amount
    
    return demand

def answer_questions(network: TrafficNetwork, demand: Dict[Tuple[str, str], float]):
    """回答软件功能部分的问题（修复版本）"""
    print("=" * 60)
    print("软件功能测试 - 问题回答")
    print("=" * 60)
    
    # 问题1: 不考虑拥堵的最快路径
    print("\n1. 不考虑拥堵，A到F的最快路径:")
    path, time = AssignmentAlgorithms.dijkstra_shortest_path(
        network, "A", "F", use_congestion=False
    )
    print(f"   路径: {' -> '.join(path)}")
    print(f"   行程时间: {time:.3f} 小时 ({time*60:.1f} 分钟)")
    
    # 问题2: 考虑拥堵的最快路径
    print("\n2. 考虑拥堵效应，A到F的最快路径:")
    aon_flow = AssignmentAlgorithms.all_or_nothing(network, demand)
    network.update_travel_time(aon_flow)
    
    path_congested, time_congested = AssignmentAlgorithms.dijkstra_shortest_path(
        network, "A", "F", use_congestion=True
    )
    print(f"   路径: {' -> '.join(path_congested)}")
    print(f"   行程时间: {time_congested:.3f} 小时 ({time_congested*60:.1f} 分钟)")
    
    # 问题3: 只考虑A到F的需求 
    print("\n3. 只考虑A到F的需求 (2000辆/小时):")
    
    
    aon_flow_single, path_details = AssignmentAlgorithms.analyze_single_od_assignment(
        network, "A", "F", 2000, "AON"
    )
    
    # 问题4: 考虑所有OD需求
    print("\n4. 考虑所有OD需求:")
    print("   总出行需求: 6000 辆/小时")
    
    # 计算不同算法的总出行时间
    print("\n   不同分配算法的总出行时间:")
    
    # AON
    aon_flow_all = AssignmentAlgorithms.all_or_nothing(network, demand)
    aon_time = AssignmentAlgorithms.calculate_total_travel_time(network, aon_flow_all)
    print(f"   AON分配: {aon_time:.2f} 车辆小时")
    
    # 增量分配
    inc_flow = AssignmentAlgorithms.incremental_assignment(network, demand, increments=10)
    inc_time = AssignmentAlgorithms.calculate_total_travel_time(network, inc_flow)
    print(f"   增量分配: {inc_time:.2f} 车辆小时")
    
    # UE分配
    ue_flow, obj_values = AssignmentAlgorithms.frank_wolfe_ue(
        network, demand, max_iterations=50, tolerance=1e-3
    )
    ue_time = AssignmentAlgorithms.calculate_total_travel_time(network, ue_flow)
    print(f"   UE分配: {ue_time:.2f} 车辆小时")
    
    print("\n   效率比较 (UE为基准):")
    print(f"   AON效率: {ue_time/aon_time*100:.1f}%")
    print(f"   增量分配效率: {ue_time/inc_time*100:.1f}%")
    
    return aon_flow_all, inc_flow, ue_flow, obj_values
def main():
    """主函数"""
    print("交通分配计算软件")
    print("=" * 40)
    
    # 检查文件是否存在
    if not os.path.exists("network.json"):
        print("错误: network.json 文件不存在！")
        return
    
    if not os.path.exists("demand.json"):
        print("错误: demand.json 文件不存在！")
        return
    
    # 加载网络和需求
    print("加载网络和需求数据...")
    network = TrafficNetwork("network.json")
    demand = load_demand("demand.json")
    
    print(f"网络节点数: {len(network.nodes)}")
    print(f"网络路段数: {len(network.link_list)}")
    print(f"OD对数量: {len(demand)}")
    print(f"总需求: {sum(demand.values())} 辆/小时")
    
    # 回答软件功能问题
    print("\n" + "=" * 60)
    print("开始交通分配计算...")
    print("=" * 60)
    
    aon_flow, inc_flow, ue_flow, obj_values = answer_questions(network, demand)
    
    # 可视化结果
    print("\n" + "=" * 60)
    print("生成可视化结果...")
    print("=" * 60)
    
    visualizer = TrafficVisualizer(network)
    
    # 创建输出目录
    os.makedirs("output", exist_ok=True)
    
    # 绘制网络结构
    print("1. 绘制网络结构图...")
    fig_network = visualizer.plot_network(
        title="交通网络结构",
        save_path="output/network_structure.png"
    )
    
    # 绘制AON分配结果
    print("2. 绘制AON分配结果...")
    fig_aon = visualizer.plot_network(
        link_flow=aon_flow,
        title="全有全无分配 (AON)",
        save_path="output/aon_assignment.png"
    )
    
    # 绘制增量分配结果
    print("3. 绘制增量分配结果...")
    fig_inc = visualizer.plot_network(
        link_flow=inc_flow,
        title="增量分配 (Incremental)",
        save_path="output/incremental_assignment.png"
    )
    
    # 绘制UE分配结果
    print("4. 绘制UE分配结果...")
    fig_ue = visualizer.plot_network(
        link_flow=ue_flow,
        title="用户均衡分配 (UE)",
        save_path="output/ue_assignment.png"
    )
    
    # 绘制流量对比
    print("5. 绘制流量对比图...")
    flow_results = {
        "AON": aon_flow,
        "增量分配": inc_flow,
        "UE分配": ue_flow
    }
    fig_comparison = visualizer.plot_flow_comparison(
        flow_results,
        save_path="output/flow_comparison.png"
    )
    
    # 绘制收敛曲线（如果UE算法收敛）
    if obj_values:
        print("6. 绘制收敛曲线...")
        fig_convergence = visualizer.plot_convergence(
            obj_values,
            save_path="output/convergence_curve.png"
        )
    
    # 保存流量结果到文件
    print("\n保存详细结果到文件...")
    with open("output/results_summary.txt", "w", encoding='utf-8') as f:
        f.write("交通分配结果汇总\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("路段流量结果:\n")
        f.write("-" * 40 + "\n")
        
        # 统计原始方向的路段
        for (start, end), link in network.links.items():
            if start < end:  # 只统计一个方向
                aon = aon_flow.get((start, end), 0)
                inc = inc_flow.get((start, end), 0)
                ue = ue_flow.get((start, end), 0)
                
                f.write(f"路段 {start}-{end}:\n")
                f.write(f"  容量: {link['capacity']} 辆/小时\n")
                f.write(f"  自由流时间: {link['free_flow_time']:.3f} 小时\n")
                f.write(f"  AON流量: {aon:.0f} 辆/小时 ({aon/link['capacity']*100:.1f}%)\n")
                f.write(f"  增量分配流量: {inc:.0f} 辆/小时 ({inc/link['capacity']*100:.1f}%)\n")
                f.write(f"  UE分配流量: {ue:.0f} 辆/小时 ({ue/link['capacity']*100:.1f}%)\n\n")
        
        # 计算总出行时间
        aon_time = AssignmentAlgorithms.calculate_total_travel_time(network, aon_flow)
        inc_time = AssignmentAlgorithms.calculate_total_travel_time(network, inc_flow)
        ue_time = AssignmentAlgorithms.calculate_total_travel_time(network, ue_flow)
        
        f.write("\n总出行时间对比:\n")
        f.write("-" * 40 + "\n")
        f.write(f"AON分配: {aon_time:.2f} 车辆小时\n")
        f.write(f"增量分配: {inc_time:.2f} 车辆小时\n")
        f.write(f"UE分配: {ue_time:.2f} 车辆小时\n")
        
        f.write(f"\nUE相对于AON的改善: {(aon_time - ue_time)/aon_time*100:.1f}%\n")
        f.write(f"UE相对于增量分配的改善: {(inc_time - ue_time)/inc_time*100:.1f}%\n")
    
    print("\n" + "=" * 60)
    print("计算完成！")
    print("=" * 60)
    print("输出文件已保存到 output/ 目录:")
    print("  - 网络结构图: output/network_structure.png")
    print("  - AON分配图: output/aon_assignment.png")
    print("  - 增量分配图: output/incremental_assignment.png")
    print("  - UE分配图: output/ue_assignment.png")
    print("  - 流量对比图: output/flow_comparison.png")
    print("  - 收敛曲线: output/convergence_curve.png")
    print("  - 结果汇总: output/results_summary.txt")

if __name__ == "__main__":
    main()