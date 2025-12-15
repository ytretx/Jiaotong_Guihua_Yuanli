import json
import math
from typing import Dict, List, Tuple, Optional
import numpy as np

class TrafficNetwork:
    """交通网络模型"""
    
    def __init__(self, network_file: str):
        """
        初始化交通网络
        
        Args:
            network_file: 网络JSON文件路径
        """
        with open(network_file, 'r') as f:
            data = json.load(f)
        
        # 读取节点信息
        self.nodes = {}
        for i, name in enumerate(data['nodes']['name']):
            self.nodes[name] = {
                'x': data['nodes']['x'][i],
                'y': data['nodes']['y'][i],
                'id': i
            }
        
        # 读取路段信息
        self.links = {}
        self.link_list = []
        
        for i, link_name in enumerate(data['links']['between']):
            start, end = link_name[0], link_name[1]
            capacity = data['links']['capacity'][i]
            speed = data['links']['speedmax'][i]
            
            # 计算路段长度
            start_x, start_y = self.nodes[start]['x'], self.nodes[start]['y']
            end_x, end_y = self.nodes[end]['x'], self.nodes[end]['y']
            length = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # 计算自由流行程时间 (小时)
            free_flow_time = length / speed
            
            self.links[(start, end)] = {
                'start': start,
                'end': end,
                'capacity': capacity,
                'speed': speed,
                'length': length,
                'free_flow_time': free_flow_time,
                'flow': 0,
                'time': free_flow_time
            }
            
            # 添加反向路段（双向道路）
            self.links[(end, start)] = {
                'start': end,
                'end': start,
                'capacity': capacity,
                'speed': speed,
                'length': length,
                'free_flow_time': free_flow_time,
                'flow': 0,
                'time': free_flow_time
            }
            
            self.link_list.append((start, end))
            self.link_list.append((end, start))
    
    def update_travel_time(self, flow: Dict[Tuple[str, str], float]) -> None:
        """
        根据流量更新路段行程时间
        
        Args:
            flow: 路段流量字典 {(start, end): flow}
        """
        for (start, end), f in flow.items():
            if (start, end) in self.links:
                link = self.links[(start, end)]
                link['flow'] = f
                # 使用BPR函数计算行程时间
                q_over_c = f / link['capacity'] if link['capacity'] > 0 else 0
                link['time'] = link['free_flow_time'] * (1 + q_over_c)**2
    
    def get_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        获取邻接矩阵
        
        Returns:
            adj_matrix: 邻接矩阵，值为自由流行程时间
            node_names: 节点名称列表
        """
        node_names = list(self.nodes.keys())
        n = len(node_names)
        adj_matrix = np.full((n, n), np.inf)
        
        # 对角线设为0
        np.fill_diagonal(adj_matrix, 0)
        
        # 填充邻接矩阵
        node_to_index = {name: i for i, name in enumerate(node_names)}
        
        for (start, end), link in self.links.items():
            i = node_to_index[start]
            j = node_to_index[end]
            adj_matrix[i][j] = link['free_flow_time']
        
        return adj_matrix, node_names
    
    def get_link_info(self, start: str, end: str) -> Optional[Dict]:
        """获取路段信息"""
        return self.links.get((start, end))
    
    def get_node_coords(self) -> Dict[str, Tuple[float, float]]:
        """获取所有节点坐标"""
        return {name: (node['x'], node['y']) for name, node in self.nodes.items()}