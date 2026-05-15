#!/usr/bin/env python3
"""
collect_sim_data.py - 使用Oracle策略批量采集仿真训练数据

这个脚本解决了OpenTrackVLA项目中缺失的关键环节：
从HM3D/MP3D场景批量生成训练数据（视频 + 动作信息）

用法:
    python collect_sim_data.py \
        --exp-config habitat-lab/habitat/config/benchmark/nav/track/track_train_stt.yaml \
        --save-path sim_data/train \
        --num-episodes 1000 \
        --seed 42

输出结构:
    sim_data/train/
        seed_42/
            <scene_id>/
                <episode_id>.mp4          # RGB视频
                <episode_id>_info.json    # 每步状态信息
                <episode_id>.json         # episode最终统计
"""

import habitat
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from habitat.config.default_structured_configs import AgentConfig
from habitat.tasks.nav.nav import NavigationEpisode
from habitat_sim.gfx import LightInfo, LightPositionModel # type: ignore
from tqdm import trange
import habitat_sim # type: ignore

import os
import os.path as osp
import imageio
import json
import argparse
import random
from typing import Optional, List

# 导入自定义模块
import evt_bench


class OracleDataCollector(AgentConfig):
    """
    使用Oracle策略采集训练数据的Agent

    与trained_agent.py中的GTBBoxAgent不同，这个类：
    1. 不需要加载任何预训练模型
    2. 使用Oracle策略（基于路径规划）来控制机器人
    3. 专注于数据采集而非评估
    """

    def __init__(self, result_path: str, sim=None):
        super().__init__()
        print("Initialize Oracle Data Collector")

        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)

        self.rgb_list = []
        self._sim = sim

        # Oracle控制器参数 (优化后)
        self.dist_thresh = 1.5  # 到目标的距离阈值 (降低以更积极追踪)
        self.safety_dist = 1.2  # 安全距离阈值 (提高以避免碰撞)
        self.turn_thresh = 0.15  # 转向角度阈值 (稍微放宽)
        self.max_forward_speed = 3.75
        self.max_tangent_speed = 1.25
        self.max_yaw_speed = 3.75

        # 追踪行为参数
        self.ideal_follow_dist = 1.8  # 理想跟踪距离
        self.min_follow_dist = 1.2    # 最小跟踪距离(低于此后退)
        self.max_follow_dist = 3.0    # 最大跟踪距离(超过此加速)

        self.reset()

    def set_sim(self, sim):
        """设置仿真器引用"""
        self._sim = sim

    def reset(self, episode: NavigationEpisode = None):
        """重置并保存上一个episode的数据"""
        if len(self.rgb_list) != 0 and episode is not None:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            output_video_path = os.path.join(save_dir, "{}.mp4".format(episode.episode_id))
            imageio.mimsave(output_video_path, self.rgb_list, fps=10)
            print(f"Saved episode video: {output_video_path}")
            self.rgb_list = []

    def _path_to_point(self, start_pos, end_pos):
        """计算从start_pos到end_pos的路径"""
        path = habitat_sim.ShortestPath()
        path.requested_start = start_pos
        path.requested_end = end_pos
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [start_pos, end_pos]
        return path.points

    def _get_angle(self, v1, v2):
        """计算两个向量之间的角度"""
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.arccos(dot)

    def _compute_oracle_action(self, robot_pos, robot_forward, human_pos):
        """
        计算Oracle动作 - 基于路径规划的专家策略

        返回: [vx, vy, wz] 速度命令
        """
        # 计算相对目标位置 (2D)
        rel_human = (human_pos - robot_pos)[[0, 2]]
        robot_forward_2d = robot_forward[[0, 2]]

        # 到人类的距离
        dist_to_human = np.linalg.norm(rel_human)

        # 安全距离检查：如果太近，需要后退
        if dist_to_human < self.min_follow_dist:
            # 计算后退速度，距离越近后退越快
            retreat_factor = (self.min_follow_dist - dist_to_human) / self.min_follow_dist
            retreat_speed = np.clip(retreat_factor * 0.8, 0.2, 0.8)  # 更积极后退
            # 同时转向面对人类
            turn_vel = self._compute_turn_speed(rel_human, robot_forward_2d)
            return [-retreat_speed, 0.0, turn_vel[2]]

        # 计算到人类的路径
        path_points = self._path_to_point(robot_pos, human_pos)

        if len(path_points) < 2:
            return [0.0, 0.0, 0.0]

        # 下一个路径点
        next_waypoint = np.array(path_points[1])

        # 计算相对目标位置 (2D)
        rel_targ = (next_waypoint - robot_pos)[[0, 2]]

        # 到下一个路径点的角度
        angle_to_target = self._get_angle(robot_forward_2d, rel_targ)
        angle_to_human = self._get_angle(robot_forward_2d, rel_human)

        # 判断是否在理想跟踪范围内
        in_ideal_range = (dist_to_human >= self.min_follow_dist and
                          dist_to_human <= self.ideal_follow_dist and
                          angle_to_human < self.turn_thresh)

        if in_ideal_range:
            # 在理想范围内，只需微调朝向
            vel = self._compute_turn_speed(rel_human, robot_forward_2d)
            return [vel[0] * 0.3, vel[1], vel[2] * 0.5]  # 减小转向幅度

        # 计算速度命令
        if dist_to_human < self.dist_thresh and angle_to_human > self.turn_thresh:
            # 距离够近但朝向不对，转向面对人类
            vel = self._compute_turn_speed(rel_human, robot_forward_2d)
        else:
            # 需要移动，计算组合速度
            vel = self._compute_combine_speed(rel_targ, robot_forward_2d, dist_to_human)

        return vel

    def _compute_turn_speed(self, rel, robot_forward):
        """计算转向速度"""
        is_left = np.cross(robot_forward, rel) > 0
        angle = self._get_angle(rel, robot_forward)
        speed_ratio = angle * 10 / self.max_yaw_speed  # ctrl_freq / 4 ≈ 10
        turn_vel = np.clip(speed_ratio, 0.1, 1.0)

        if is_left:
            return [0.0, 0.0, -turn_vel * self.max_yaw_speed * 0.1]
        else:
            return [0.0, 0.0, turn_vel * self.max_yaw_speed * 0.1]

    def _compute_combine_speed(self, rel_targ, robot_forward, dist_to_human):
        """计算组合速度（前进 + 侧向 + 转向）"""
        # 归一化机器人前向
        robot_forward_norm = robot_forward / (np.linalg.norm(robot_forward) + 1e-8)
        robot_right = np.array([-robot_forward_norm[1], robot_forward_norm[0]])

        # 变换矩阵：世界坐标 -> 机器人坐标
        transform_matrix = np.array([robot_right, robot_forward_norm]).T

        # 根据距离调整速度增益 - 使用新的跟踪距离参数
        if dist_to_human < self.ideal_follow_dist:
            # 接近理想距离，减速
            speed_scale = 0.5
        elif dist_to_human < self.max_follow_dist:
            # 在理想和最大距离之间，正常速度
            speed_scale = 0.75
        else:
            # 超过最大跟踪距离，全速追踪
            speed_scale = 1.0

        # 在机器人坐标系中的相对位置
        rel_robot = np.dot(transform_matrix, rel_targ)

        # 计算速度
        forward_speed = max(rel_robot[1], 0) * 10  # ctrl_freq / 4
        tangent_speed = rel_robot[0] * 10

        # 限制速度
        forward_ratio = min(abs(self.max_forward_speed / (forward_speed + 1e-8)), speed_scale)
        tangent_ratio = min(abs(self.max_tangent_speed / (tangent_speed + 1e-8)), 1.0)
        ratio = min(forward_ratio, tangent_ratio)

        # 提高速度增益 (从0.3提高到0.5-0.7)
        speed_gain = 0.5 if dist_to_human < self.ideal_follow_dist else 0.7
        vx = forward_speed * ratio / self.max_forward_speed * self.max_forward_speed * speed_gain
        vy = -tangent_speed * ratio / self.max_tangent_speed * self.max_tangent_speed * speed_gain * 0.8

        # 计算转向速度
        is_left = np.cross(robot_forward, rel_targ) > 0
        angle = self._get_angle(rel_targ, robot_forward)
        turn_ratio = angle * 10 / self.max_yaw_speed
        wz = np.clip(turn_ratio, 0, 1) * self.max_yaw_speed * speed_gain
        if is_left:
            wz = -wz

        return [float(vx), float(vy), float(wz)]

    def act(self, observations, sim, episode_id, instruction: Optional[str] = None):
        """
        根据观测计算Oracle动作

        Args:
            observations: 仿真器观测
            sim: 仿真器实例
            episode_id: 当前episode ID
            instruction: 指令文本（用于记录）

        Returns:
            [vx, vy, wz]: 速度命令
        """
        self._sim = sim

        # 获取RGB图像
        rgb = observations["agent_1_articulated_agent_jaw_rgb"]
        rgb_ = rgb[:, :, :3]

        # 获取机器人和人类位置
        robot_agent = sim.agents_mgr[1].articulated_agent
        human_agent = sim.agents_mgr[0].articulated_agent

        robot_pos = np.array(robot_agent.base_pos)
        human_pos = np.array(human_agent.base_pos)

        # 获取机器人前向方向
        base_T = robot_agent.base_transformation
        forward = np.array([1.0, 0, 0])
        robot_forward = np.array(base_T.transform_vector(forward))

        # 计算Oracle动作
        action = self._compute_oracle_action(robot_pos, robot_forward, human_pos)

        # 添加一些随机扰动使数据更多样化
        noise_scale = 0.05
        action[0] += np.random.normal(0, noise_scale)
        action[1] += np.random.normal(0, noise_scale)
        action[2] += np.random.normal(0, noise_scale * 0.5)

        # 保存帧
        self.rgb_list.append(rgb_)  # Habitat outputs RGB, keep as-is

        return action


def collect_data(config, dataset_split, save_path: str, seed: int) -> dict:
    """
    主数据采集循环

    Args:
        config: Habitat配置
        dataset_split: 数据集分割
        save_path: 保存路径
        seed: 随机种子

    Returns:
        统计信息字典
    """
    # 创建保存目录
    seed_path = os.path.join(save_path, f"seed_{seed}")
    os.makedirs(seed_path, exist_ok=True)

    # 初始化采集器
    collector = OracleDataCollector(seed_path)

    stats = {
        "total_episodes": 0,
        "successful_episodes": 0,
        "collision_episodes": 0,
        "lost_episodes": 0,
    }

    with habitat.TrackEnv(config=config, dataset=dataset_split) as env:
        sim = env.sim
        collector.set_sim(sim)
        collector.reset()

        num_episodes = len(env.episodes)
        print(f"Total episodes to collect: {num_episodes}")

        for ep_idx in trange(num_episodes, desc="Collecting data"):
            obs = env.reset()

            # 设置光照
            light_setup = [
                LightInfo(vector=[10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[-10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, 10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, -10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
            ]
            sim.set_light_setup(light_setup)

            # 获取指令
            try:
                instruction = env.current_episode.info.get('instruction', "Follow the target person.")
            except Exception:
                instruction = "Follow the target person."

            record_infos = []
            result = {}

            human_agent = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent

            iter_step = 0
            followed_step = 0
            too_far_count = 0
            status = 'Normal'
            finished = False

            while not env.episode_over:
                record_info = {}

                # 获取观测
                obs = sim.get_sensor_observations()
                detector = env.task._get_observations(env.current_episode)

                # 使用Oracle策略计算动作
                action = collector.act(obs, sim, env.current_episode.episode_id, instruction)

                # 构建动作字典
                action_dict = {
                    "action": (
                        "agent_0_humanoid_navigate_action",
                        "agent_1_base_velocity",
                        "agent_2_oracle_nav_randcoord_action_obstacle",
                        "agent_3_oracle_nav_randcoord_action_obstacle",
                        "agent_4_oracle_nav_randcoord_action_obstacle",
                        "agent_5_oracle_nav_randcoord_action_obstacle",
                    ),
                    "action_args": {
                        "agent_1_base_vel": action
                    }
                }

                iter_step += 1
                env.step(action_dict)

                # 获取指标
                info = env.get_metrics()

                # 记录跟踪状态
                if info.get('human_following', 0) == 1.0:
                    followed_step += 1
                    too_far_count = 0

                # 检查是否太远
                dis_to_human = np.linalg.norm(robot_agent.base_pos - human_agent.base_pos)
                if dis_to_human > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        status = 'Lost'
                        finished = False
                        break

                # 记录步骤信息
                record_info["step"] = iter_step
                record_info["dis_to_human"] = float(dis_to_human)
                record_info["facing"] = info.get('human_following', 0)
                record_info["base_velocity"] = action
                record_infos.append(record_info)

                # 检查碰撞
                if info.get('human_collision', 0) == 1.0:
                    status = 'Collision'
                    finished = False
                    break

            # Episode结束处理
            if env.episode_over:
                finished = True
                status = 'Success' if followed_step / max(iter_step, 1) > 0.5 else 'Normal'

            info = env.get_metrics()

            # 保存数据
            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(seed_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)

            # 保存步骤信息
            with open(os.path.join(save_dir, f"{env.current_episode.episode_id}_info.json"), "w") as f:
                json.dump(record_infos, f, indent=2)

            # 保存episode统计
            result['finish'] = finished
            result['status'] = status
            result['success'] = 1.0 if status == 'Success' else 0.0
            result['following_rate'] = followed_step / max(iter_step, 1)
            result['following_step'] = followed_step
            result['total_step'] = iter_step
            result['collision'] = info.get('human_collision', 0)
            result['instruction'] = instruction

            with open(os.path.join(save_dir, f"{env.current_episode.episode_id}.json"), "w") as f:
                json.dump(result, f, indent=2)

            # 重置采集器（保存视频）
            collector.reset(env.current_episode)

            # 更新统计
            stats["total_episodes"] += 1
            if status == 'Success':
                stats["successful_episodes"] += 1
            elif status == 'Collision':
                stats["collision_episodes"] += 1
            elif status == 'Lost':
                stats["lost_episodes"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Collect simulation training data using Oracle policy")
    parser.add_argument(
        "--exp-config",
        type=str,
        default="habitat-lab/habitat/config/benchmark/nav/track/track_train_stt.yaml",
        help="Path to Habitat config yaml",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="sim_data/train",
        help="Path to save collected data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--split-id",
        type=int,
        default=0,
        help="Dataset split ID (for parallel collection)",
    )
    parser.add_argument(
        "--split-num",
        type=int,
        default=1,
        help="Total number of splits",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to collect (None = all)",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 加载配置
    config = habitat.get_config(args.exp_config, args.opts)

    # 加载数据集
    from habitat.datasets import make_dataset
    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)

    # 获取数据集分割
    if args.split_num > 1:
        dataset_split = dataset.get_splits(args.split_num)[args.split_id]
    else:
        dataset_split = dataset

    # 限制episode数量
    if args.num_episodes is not None and args.num_episodes < len(dataset_split.episodes):
        dataset_split.episodes = dataset_split.episodes[:args.num_episodes]

    print(f"Configuration:")
    print(f"  Config: {args.exp_config}")
    print(f"  Save path: {args.save_path}")
    print(f"  Seed: {args.seed}")
    print(f"  Split: {args.split_id}/{args.split_num}")
    print(f"  Episodes: {len(dataset_split.episodes)}")

    # 开始采集
    stats = collect_data(config, dataset_split, args.save_path, args.seed)

    # 打印统计
    print("\n" + "="*50)
    print("Collection Statistics:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Successful: {stats['successful_episodes']}")
    print(f"  Collision: {stats['collision_episodes']}")
    print(f"  Lost: {stats['lost_episodes']}")
    print("="*50)


if __name__ == "__main__":
    main()
