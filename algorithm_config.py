"""
算法配置管理模块
支持降噪和超分辨率算法的启用/禁用、自定义名称
配置持久化保存为 JSON 文件
"""

import json
import os
from typing import List, Dict, Tuple

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'algorithm_config.json')

# 默认降噪算法配置
DEFAULT_DENOISE_ALGORITHMS = [
    {"key": "hybrid", "name": "Hybrid (Recommended)", "enabled": True},
    {"key": "bm3d", "name": "BM3D (Block-Matching 3D)", "enabled": True},
    {"key": "anisotropic", "name": "Anisotropic Diffusion", "enabled": True},
    {"key": "iterative", "name": "Iterative Reconstruction", "enabled": True},
    {"key": "nlm", "name": "Non-local Means", "enabled": True},
    {"key": "bilateral", "name": "Bilateral Filter", "enabled": True},
    {"key": "wavelet", "name": "Wavelet", "enabled": True},
    {"key": "gaussian", "name": "Gaussian Filter", "enabled": True},
]

# 默认超分辨率算法配置
DEFAULT_SR_ALGORITHMS = [
    {"key": "bicubic", "name": "双三次插值 (Bicubic)", "enabled": True},
    {"key": "lanczos", "name": "兰索斯插值 (Lanczos) - 推荐", "enabled": True},
    {"key": "edge_preserving", "name": "保边增强 (Edge Preserving)", "enabled": True},
]


def load_config() -> Dict:
    """加载配置文件，不存在时创建默认配置。"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 确保配置结构完整
                if 'denoise' not in config:
                    config['denoise'] = {'algorithms': DEFAULT_DENOISE_ALGORITHMS}
                if 'super_resolution' not in config:
                    config['super_resolution'] = {'algorithms': DEFAULT_SR_ALGORITHMS}
                return config
        except (json.JSONDecodeError, IOError) as e:
            print(f"加载配置文件失败：{e}，使用默认配置")

    # 创建默认配置
    config = {
        'denoise': {'algorithms': DEFAULT_DENOISE_ALGORITHMS},
        'super_resolution': {'algorithms': DEFAULT_SR_ALGORITHMS}
    }
    save_config(config)
    return config


def save_config(config: Dict) -> bool:
    """保存配置到文件。"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except IOError as e:
        print(f"保存配置文件失败：{e}")
        return False


def get_denoise_algorithms(include_disabled: bool = False) -> List[Dict]:
    """
    获取降噪算法列表。

    Args:
        include_disabled: 是否包含禁用的算法

    Returns:
        算法列表，每个元素包含 key 和 name
    """
    config = load_config()
    algorithms = config.get('denoise', {}).get('algorithms', DEFAULT_DENOISE_ALGORITHMS)

    if include_disabled:
        return [{"key": a["key"], "name": a["name"]} for a in algorithms]
    else:
        return [{"key": a["key"], "name": a["name"]} for a in algorithms if a.get("enabled", True)]


def get_sr_algorithms(include_disabled: bool = False) -> List[Dict]:
    """
    获取超分辨率算法列表。

    Args:
        include_disabled: 是否包含禁用的算法

    Returns:
        算法列表，每个元素包含 key 和 name
    """
    config = load_config()
    algorithms = config.get('super_resolution', {}).get('algorithms', DEFAULT_SR_ALGORITHMS)

    if include_disabled:
        return [{"key": a["key"], "name": a["name"]} for a in algorithms]
    else:
        return [{"key": a["key"], "name": a["name"]} for a in algorithms if a.get("enabled", True)]


def update_algorithm(algo_type: str, key: str, name: str = None, enabled: bool = None) -> bool:
    """
    更新算法配置。

    Args:
        algo_type: 算法类型 ('denoise' 或 'super_resolution')
        key: 算法键值
        name: 新的显示名称（可选）
        enabled: 新的启用状态（可选）

    Returns:
        bool: 是否成功
    """
    config = load_config()
    algorithms = config.get(algo_type, {}).get('algorithms', [])

    for algo in algorithms:
        if algo['key'] == key:
            if name is not None:
                algo['name'] = name
            if enabled is not None:
                algo['enabled'] = enabled
            return save_config(config)

    return False


def add_algorithm(algo_type: str, key: str, name: str, enabled: bool = True) -> bool:
    """
    添加新算法配置。

    Args:
        algo_type: 算法类型 ('denoise' 或 'super_resolution')
        key: 算法键值
        name: 显示名称
        enabled: 是否启用

    Returns:
        bool: 是否成功
    """
    config = load_config()

    if algo_type not in config:
        config[algo_type] = {'algorithms': []}

    algorithms = config[algo_type].get('algorithms', [])

    # 检查是否已存在
    for algo in algorithms:
        if algo['key'] == key:
            return False

    algorithms.append({"key": key, "name": name, "enabled": enabled})
    config[algo_type]['algorithms'] = algorithms
    return save_config(config)


def delete_algorithm(algo_type: str, key: str) -> bool:
    """
    删除算法配置（仅从配置中移除，不删除代码）。

    Args:
        algo_type: 算法类型 ('denoise' 或 'super_resolution')
        key: 算法键值

    Returns:
        bool: 是否成功
    """
    config = load_config()
    algorithms = config.get(algo_type, {}).get('algorithms', [])

    original_count = len(algorithms)
    algorithms = [a for a in algorithms if a['key'] != key]

    if len(algorithms) < original_count:
        config[algo_type]['algorithms'] = algorithms
        return save_config(config)

    return False


def get_algorithm_config(algo_type: str) -> List[Dict]:
    """
    获取完整的算法配置（包括 enabled 状态）。

    Args:
        algo_type: 算法类型 ('denoise' 或 'super_resolution')

    Returns:
        完整的算法配置列表
    """
    config = load_config()
    return config.get(algo_type, {}).get('algorithms', [])


def reset_to_defaults(algo_type: str = None) -> bool:
    """
    恢复默认配置。

    Args:
        algo_type: 算法类型，None 表示重置所有

    Returns:
        bool: 是否成功
    """
    config = load_config()

    if algo_type is None or algo_type == 'denoise':
        config['denoise'] = {'algorithms': DEFAULT_DENOISE_ALGORITHMS}

    if algo_type is None or algo_type == 'super_resolution':
        config['super_resolution'] = {'algorithms': DEFAULT_SR_ALGORITHMS}

    return save_config(config)
