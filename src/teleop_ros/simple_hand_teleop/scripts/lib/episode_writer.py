import os
import time
import json

import numpy as np
import cv2
import open3d as o3d

from typing import List, Dict, Optional

class EpisodeWriter:
    def __init__(self, task_dir: str) -> None:
        super().__init__()

        print("==> EpisodeWriter initializing...")

        self.task_dir = task_dir
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir) if "episode_" in episode_dir]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = -1 if episode_last is None else int(episode_last.split('_')[-1])
            print(f"==> task_dir directory already exist, now self.episode_id is: {self.episode_id}")
        else:
            os.makedirs(self.task_dir)
            self.episode_id = -1
            print(f"==> episode directory does not exist, now create one.")

        print("==> EpisodeWriter initialized successfully.")
    
    def create_episode(self) -> None:
        """
        Create a new episode, each episode needs to specify the episode_id.
            text: Text descriptions of operation goals, steps, etc. The text description of each episode is the same.
            goal: operation goal
            desc: description
            steps: operation steps
        """
        print("==> Starting record a episode...")

        self.item_id = -1
        self.episode_data = []
        self.episode_id = self.episode_id + 1
        
        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, "colors")
        self.depth_dir = os.path.join(self.episode_dir, "depths")
        self.pointcloud_dir = os.path.join(self.episode_dir, "pointclouds")
        self.json_path = os.path.join(self.episode_dir, "data.json")
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.pointcloud_dir, exist_ok=True)

        print("==> Starting record a episode successfully...")

    def add_item(self, colors: Dict[str, np.ndarray], 
                 states: Dict[str, Dict[str, List[float]]],
                 actions: Dict[str, Dict[str, List[float]]],
                 depths: Optional[Dict[str, np.ndarray]] = None,
                 pointclouds: Optional[Dict[str, np.ndarray]] = None,
                 log: bool = True
                 ) -> None:
        self.item_id += 1
        item_data = {
            "idx": self.item_id,
            "colors": {},       
            "depths": {},
            "pointclouds": {},
            "states": {},
            "actions": {},
        }

        # save colors
        for key, color in colors.items():
            color_name = f"{str(self.item_id).zfill(6)}_{key}.jpg"
            cv2.imwrite(os.path.join(self.color_dir, color_name), color)
            item_data["colors"][key] = os.path.join("colors", color_name)

        # save depths
        if depths is not None:
            for key, depth in depths.items():
                depth_name = f"{str(self.item_id).zfill(6)}_{key}.png"
                cv2.imwrite(os.path.join(self.depth_dir, depth_name), depth)
                item_data["depths"][key] = os.path.join("depths", depth_name)
        
        # save pointclouds
        if pointclouds is not None:
            for key, pointcloud in pointclouds.items():
                pointcloud_name = f"{str(self.item_id).zfill(6)}_{key}.pcd"
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pointcloud)
                o3d.io.write_point_cloud(os.path.join(self.pointcloud_dir, pointcloud_name), pcd)
                item_data["pointclouds"][key] = os.path.join("pointclouds", pointcloud_name)
        
        item_data["states"] = states
        item_data["actions"] = actions

        self.episode_data.append(item_data)

        if log:
            current_record_time = time.time()
            print(f"==> episode_id:{self.episode_id}  item_id:{self.item_id}  current_time:{current_record_time}")
    
    def save_episode(self):
        print("==> Ending record a episode...Save the episode...")

        data = dict()

        data["data"] = self.episode_data

        with open(self.json_path, 'w', encoding="utf-8") as jsonf:
            jsonf.write(json.dumps(data, indent=4, ensure_ascii=False))
        
        print("==> Save the episode successfully...")