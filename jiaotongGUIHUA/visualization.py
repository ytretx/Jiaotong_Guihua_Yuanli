import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问

class TrafficVisualizer:
    """交通网络可视化"""
    
    def __init__(self, network):
        """
        初始化可视化器
        
        Args:
            network: TrafficNetwork对象
        """
        self.network = network
    
    def plot_network(self, link_flow: Dict[Tuple[str, str], float] = None, 
                    title: str = "交通网络", save_path: str = None) -> plt.Figure:
        """
        绘制交通网络
        
        Args:
            link_flow: 路段流量字典（如果提供，则显示流量）
            title: 图标题
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 获取节点坐标
        node_coords = self.network.get_node_coords()
        
        # 绘制节点
        for node_name, (x, y) in node_coords.items():
            ax.plot(x, y, 'o', markersize=12, color='blue', alpha=0.7)
            ax.text(x, y, f'  {node_name}', fontsize=12, fontweight='bold', 
                   verticalalignment='center', color='black')
        
        # 绘制路段
        max_flow = 0
        if link_flow:
            max_flow = max(link_flow.values()) if link_flow.values() else 0
        
        for (start, end), link in self.network.links.items():
            # 只绘制原始的路段（避免重复绘制双向路段）
            if (start, end) in self.network.links and start < end:
                x1, y1 = node_coords[start]
                x2, y2 = node_coords[end]
                
                # 计算中点用于标注
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                # 计算线条宽度（如果提供流量）
                linewidth = 1
                if link_flow and max_flow > 0:
                    flow = link_flow.get((start, end), 0)
                    linewidth = max(1, 3 * flow / max_flow)
                
                # 绘制路段
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=linewidth, alpha=0.7)
                
                # 标注流量（如果提供）
                if link_flow:
                    flow = link_flow.get((start, end), 0)
                    if flow > 0:
                        # 在路段旁边标注流量
                        offset_x = (y2 - y1) * 0.1  # 垂直于路段的偏移
                        offset_y = -(x2 - x1) * 0.1
                        ax.text(mid_x + offset_x, mid_y + offset_y, 
                               f'{flow:.0f}', fontsize=10, 
                               bbox=dict(boxstyle="round,pad=0.3", 
                                        facecolor="lightyellow", alpha=0.8))
                
                # 标注路段属性
                prop_text = f"cap={link['capacity']}\nv={link['speed']}"
                ax.text(mid_x, mid_y, prop_text, fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.2", 
                                facecolor="lightblue", alpha=0.6))
        
        # 设置图形属性
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X坐标 (km)', fontsize=12)
        ax.set_ylabel('Y坐标 (km)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_flow_comparison(self, flow_results: Dict[str, Dict[Tuple[str, str], float]], 
                           save_path: str = None) -> plt.Figure:
        """
        绘制不同算法流量对比
        
        Args:
            flow_results: 算法名到流量字典的映射
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(len(flow_results), 1, figsize=(14, 4*len(flow_results)))
        
        if len(flow_results) == 1:
            axes = [axes]
        
        for idx, (algo_name, link_flow) in enumerate(flow_results.items()):
            ax = axes[idx]
            
            # 获取路段名称和流量
            link_names = []
            flows = []
            
            for (start, end), link in self.network.links.items():
                # 只统计原始方向的路段
                if start < end:
                    link_names.append(f"{start}-{end}")
                    flows.append(link_flow.get((start, end), 0))
            
            # 绘制条形图
            x_pos = np.arange(len(link_names))
            bars = ax.bar(x_pos, flows, alpha=0.7, color='steelblue')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height:.0f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('路段', fontsize=12)
            ax.set_ylabel('流量 (辆/小时)', fontsize=12)
            ax.set_title(f'{algo_name}分配结果', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(link_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_convergence(self, objective_values: List[float], save_path: str = None) -> plt.Figure:
        """
        绘制Frank-Wolfe算法收敛曲线
        
        Args:
            objective_values: 目标函数值列表
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(1, len(objective_values) + 1)
        ax.plot(iterations, objective_values, 'b-o', linewidth=2, markersize=6)
        
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel('总出行时间 (车辆小时)', fontsize=12)
        ax.set_title('Frank-Wolfe算法收敛过程', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, len(objective_values))
        
        # 添加收敛标记
        if len(objective_values) > 10:
            final_value = objective_values[-1]
            ax.axhline(y=final_value, color='r', linestyle='--', alpha=0.7, 
                      label=f'收敛值: {final_value:.2f}')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig