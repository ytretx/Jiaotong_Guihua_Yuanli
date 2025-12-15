import numpy as np
from typing import Dict, List, Tuple, Set
import heapq
from scipy.optimize import minimize_scalar

class AssignmentAlgorithms:
    """交通分配算法实现"""
    
    @staticmethod
    def dijkstra_shortest_path(network, start: str, end: str, use_congestion: bool = False) -> Tuple[List[str], float]:
        """
        Dijkstra算法寻找最短路径
        
        Args:
            network: TrafficNetwork对象
            start: 起点
            end: 终点
            use_congestion: 是否考虑拥堵（True使用当前时间，False使用自由流时间）
            
        Returns:
            path: 最短路径节点列表
            total_time: 总行程时间
        """
        # 初始化
        dist = {node: float('inf') for node in network.nodes}
        prev = {node: None for node in network.nodes}
        dist[start] = 0
        
        # 优先队列
        pq = [(0, start)]
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            # 如果找到终点
            if current_node == end:
                break
            
            # 如果当前距离大于记录的距离，跳过
            if current_dist > dist[current_node]:
                continue
            
            # 遍历邻居
            for (s, e), link in network.links.items():
                if s == current_node:
                    # 选择时间计算方式
                    travel_time = link['time'] if use_congestion else link['free_flow_time']
                    
                    new_dist = current_dist + travel_time
                    
                    if new_dist < dist[e]:
                        dist[e] = new_dist
                        prev[e] = current_node
                        heapq.heappush(pq, (new_dist, e))
        
        # 重建路径
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()
        
        return path, dist[end]
    
    @staticmethod
    def all_or_nothing(network, demand: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
        """
        全有全无分配算法
        
        Args:
            network: TrafficNetwork对象
            demand: OD需求字典 {(origin, destination): demand}
            
        Returns:
            路段流量字典
        """
        # 初始化流量为零
        link_flow = {link_id: 0.0 for link_id in network.links.keys()}
        
        # 为每个OD对分配流量
        for (origin, destination), od_demand in demand.items():
            # 寻找最短路径（自由流状态）
            path, _ = AssignmentAlgorithms.dijkstra_shortest_path(
                network, origin, destination, use_congestion=False
            )
            
            # 将OD需求加载到路径的所有路段上
            for i in range(len(path) - 1):
                start, end = path[i], path[i + 1]
                link_flow[(start, end)] += od_demand
        
        return link_flow
    
    @staticmethod
    def incremental_assignment(network, demand: Dict[Tuple[str, str], float], 
                              increments: int = 10) -> Dict[Tuple[str, str], float]:
        """
        增量分配算法
        
        Args:
            network: TrafficNetwork对象
            demand: OD需求字典
            increments: 增量份数
            
        Returns:
            路段流量字典
        """
        # 初始化流量为零
        link_flow = {link_id: 0.0 for link_id in network.links.keys()}
        
        # 将每个OD对的需求分成若干份
        incremental_demand = {od: demand[od] / increments for od in demand}
        
        # 进行多轮分配
        for increment in range(increments):
            # 基于当前流量更新行程时间
            network.update_travel_time(link_flow)
            
            # 为每个OD对的增量需求进行全有全无分配
            incremental_flow = AssignmentAlgorithms.all_or_nothing(network, incremental_demand)
            
            # 累加流量
            for link_id in link_flow:
                link_flow[link_id] += incremental_flow[link_id]
        
        return link_flow
    
    @staticmethod
    def frank_wolfe_ue(network, demand: Dict[Tuple[str, str], float], 
                      max_iterations: int = 100, tolerance: float = 1e-4) -> Tuple[Dict[Tuple[str, str], float], List[float]]:
        """
        Frank-Wolfe算法实现用户均衡分配
        
        Args:
            network: TrafficNetwork对象
            demand: OD需求字典
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            
        Returns:
            路段流量字典
            每次迭代的目标函数值列表
        """
        n_links = len(network.link_list)
        link_index = {link_id: i for i, link_id in enumerate(network.link_list)}
        
        # 初始化流量（使用AON结果作为起点）
        x_current = np.zeros(n_links)
        initial_flow = AssignmentAlgorithms.all_or_nothing(network, demand)
        for link_id, flow in initial_flow.items():
            if link_id in link_index:
                x_current[link_index[link_id]] = flow
        
        # 存储目标函数值
        objective_values = []
        
        for iteration in range(max_iterations):
            # 基于当前流量计算行程时间
            current_flow_dict = {link_id: x_current[i] for i, link_id in enumerate(network.link_list)}
            network.update_travel_time(current_flow_dict)
            
            # 计算当前目标函数值（总出行时间）
            total_time = 0
            for i, link_id in enumerate(network.link_list):
                link = network.links[link_id]
                q = x_current[i]
                t0 = link['free_flow_time']
                cap = link['capacity']
                # 计算积分形式的目标函数 ∫₀^q t(x)dx
                if cap > 0:
                    total_time += t0 * (q + q**2/(2*cap) + q**3/(3*cap**2))
            objective_values.append(total_time)
            
            # 进行辅助分配（基于当前时间的最短路径分配）
            auxiliary_flow = np.zeros(n_links)
            for (origin, destination), od_demand in demand.items():
                path, _ = AssignmentAlgorithms.dijkstra_shortest_path(
                    network, origin, destination, use_congestion=True
                )
                
                for i in range(len(path) - 1):
                    start, end = path[i], path[i + 1]
                    idx = link_index[(start, end)]
                    auxiliary_flow[idx] += od_demand
            
            # 计算下降方向
            direction = auxiliary_flow - x_current
            
            # 使用黄金分割法寻找最优步长
            def objective_function(alpha):
                x_test = x_current + alpha * direction
                total = 0
                for i, link_id in enumerate(network.link_list):
                    link = network.links[link_id]
                    q = x_test[i]
                    t0 = link['free_flow_time']
                    cap = link['capacity']
                    if cap > 0:
                        total += t0 * (q + q**2/(2*cap) + q**3/(3*cap**2))
                return total
            
            result = minimize_scalar(objective_function, bounds=(0, 1), method='bounded')
            optimal_alpha = result.x
            
            # 更新流量
            x_new = x_current + optimal_alpha * direction
            
            # 检查收敛性
            if np.linalg.norm(x_new - x_current) < tolerance:
                x_current = x_new
                print(f"Frank-Wolfe算法在第{iteration+1}次迭代后收敛")
                break
            
            x_current = x_new
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration+1}: 总出行时间 = {total_time:.2f}")
        
        # 转换为字典格式
        final_flow = {}
        for i, link_id in enumerate(network.link_list):
            final_flow[link_id] = x_current[i]
        
        return final_flow, objective_values
    
    @staticmethod
    def calculate_total_travel_time(network, link_flow: Dict[Tuple[str, str], float]) -> float:
        """
        计算路网总出行时间
        
        Args:
            network: TrafficNetwork对象
            link_flow: 路段流量字典
            
        Returns:
            总出行时间（车辆小时）
        """
        total_time = 0
        
        for (start, end), flow in link_flow.items():
            link = network.links.get((start, end))
            if link and flow > 0:
                # 使用BPR函数计算行程时间
                t0 = link['free_flow_time']
                cap = link['capacity']
                q_over_c = flow / cap if cap > 0 else 0
                travel_time = t0 * (1 + q_over_c)**2
                total_time += flow * travel_time
        
        return total_time
    
    
    #路径搜索
    @staticmethod
    def analyze_single_od_assignment(network, origin: str, destination: str, demand: float, 
                                    algorithm_name: str = "AON"):
        """
        分析单个OD对的分配结果
        
        Args:
            network: TrafficNetwork对象
            origin: 起点
            destination: 终点
            demand: 交通需求
            algorithm_name: 算法名称
        """
        print(f"\n{'='*60}")
        print(f"单个OD对分析: {origin} → {destination} (需求: {demand} 辆/小时)")
        print(f"使用算法: {algorithm_name}")
        print(f"{'='*60}")
        
        # 创建单个OD对的需求
        single_demand = {(origin, destination): demand}
        
        # 根据算法选择分配方法
        if algorithm_name == "AON":
            link_flow = AssignmentAlgorithms.all_or_nothing(network, single_demand)
        elif algorithm_name == "增量分配":
            link_flow = AssignmentAlgorithms.incremental_assignment(network, single_demand, increments=10)
        elif algorithm_name == "UE":
            link_flow, _ = AssignmentAlgorithms.frank_wolfe_ue(network, single_demand, max_iterations=50, tolerance=1e-3)
        else:
            print(f"未知算法: {algorithm_name}")
            return None, None
        
        # 更新网络行程时间
        network.update_travel_time(link_flow)
        
        # 1. 显示各路段流量
        print("\n1. 各路段的流量:")
        print("-" * 40)
        total_flow = 0
        
        # 只显示原始方向的路段
        for (start, end), link in network.links.items():
            if start < end:  # 避免重复显示双向路段
                flow = link_flow.get((start, end), 0)
                if flow > 0:
                    capacity_ratio = flow / link['capacity'] * 100 if link['capacity'] > 0 else 0
                    print(f"   路段 {start}-{end}: {flow:.0f} 辆/小时 ({capacity_ratio:.1f}% 容量)")
                    total_flow += flow
        
        print(f"\n   总分配流量: {total_flow:.0f} 辆/小时")
        
        # 2. 找出所有被使用的路径
        print("\n2. 寻找被使用的路径:")
        print("-" * 40)
        
        def find_all_paths(current, target, visited, path, all_paths, flow_dict):
            """DFS寻找所有路径"""
            if current == target:
                all_paths.append(path.copy())
                return
            
            visited.add(current)
            
            # 遍历所有邻居节点（流量大于0的路段）
            for (start, end), flow in flow_dict.items():
                if start == current and flow > 0 and end not in visited:
                    path.append(end)
                    find_all_paths(end, target, visited, path, all_paths, flow_dict)
                    path.pop()
            
            visited.remove(current)
        
        # 执行路径搜索
        all_paths = []
        find_all_paths(origin, destination, set(), [origin], all_paths, link_flow)
        
        if not all_paths:
            print("    警告: 未找到任何路径！检查网络连接性。")
            return link_flow, None
        
        print(f"    发现 {len(all_paths)} 条被使用的路径:")
        
        # 3. 计算每条路径的行程时间和流量分布
        print("\n3. 各路径详情:")
        print("-" * 40)
        
        path_details = []
        
        for i, path in enumerate(all_paths):
            # 计算路径总行程时间
            total_time = 0
            path_links = []
            
            for j in range(len(path) - 1):
                start, end = path[j], path[j + 1]
                link = network.links.get((start, end))
                if link:
                    total_time += link['time']
                    path_links.append((start, end))
            
            # 估算路径流量
            path_flow = demand / len(all_paths) if len(all_paths) > 0 else demand
            
            path_details.append({
                'index': i + 1,
                'path': path,
                'links': path_links,
                'time': total_time,
                'flow': path_flow
            })
            
            print(f"    路径{i+1}: {' → '.join(path)}")
            print(f"        行程时间: {total_time:.3f} 小时 ({total_time*60:.1f} 分钟)")
            print(f"        估计流量: {path_flow:.0f} 辆/小时")
        
        # 4. 检查行程时间是否相等
        print("\n4. 行程时间相等性分析:")
        print("-" * 40)
        
        if len(path_details) > 1:
            times = [p['time'] for p in path_details]
            min_time = min(times)
            max_time = max(times)
            
            # 计算相对误差
            max_error = (max_time - min_time) / min_time * 100 if min_time > 0 else 0
            
            print(f"    最短路径时间: {min_time:.3f} 小时")
            print(f"    最长路径时间: {max_time:.3f} 小时")
            print(f"    最大时间差异: {max_error:.2f}%")
            
            if max_error < 5.0:  # 5%以内的差异认为相等
                print(f"    ✅ 结论: 各路径行程时间基本相等（差异 < 5%）")
                print(f"    符合用户均衡(UE)的基本原理")
            else:
                print(f"    ⚠️  结论: 各路径行程时间存在明显差异")
                print(f"    可能未达到完全的用户均衡状态")
        else:
            print(f"    只有一条路径，无需比较行程时间相等性")
        
        return link_flow, path_details