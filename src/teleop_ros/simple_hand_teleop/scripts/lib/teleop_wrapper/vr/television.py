from vuer import Vuer, VuerSession
from vuer.events import ClientEvent
from vuer.schemas import Hands, ImageBackground
from asyncio import sleep

from multiprocessing import Process, Lock
from multiprocessing.managers import SharedMemoryManager, SyncManager

import numpy as np
import math

from typing import Optional

class TeleVision:
    def __init__(self, display_image_array: np.ndarray) -> None:
        super().__init__()
        
        self.manager = SyncManager()
        self.manager.start()
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self.lock = Lock()

        self.is_left_hand_initialized = self.manager.Value('b', False)
        self.left_hand_shm = self.shm_manager.SharedMemory(size=16 * np.float64().nbytes)
        self.left_hand_array = np.ndarray((16,), dtype=np.float64, buffer=self.left_hand_shm.buf)
        self.is_right_hand_initialized = self.manager.Value('b', False)
        self.right_hand_shm = self.shm_manager.SharedMemory(size=16 * np.float64().nbytes)
        self.right_hand_array = np.ndarray((16,), dtype=np.float64, buffer=self.right_hand_shm.buf)
        self.is_left_landmarks_initialized = self.manager.Value('b', False)
        self.left_landmarks_shm = self.shm_manager.SharedMemory(size=25 * 3 * np.float64().nbytes)
        self.left_landmarks_array = np.ndarray((25, 3), dtype=np.float64, buffer=self.left_landmarks_shm.buf)
        self.is_right_landmarks_initialized = self.manager.Value('b', False)
        self.right_landmarks_shm = self.shm_manager.SharedMemory(size=25 * 3 * np.float64().nbytes)
        self.right_landmarks_array = np.ndarray((25, 3), dtype=np.float64, buffer=self.right_landmarks_shm.buf)

        self.display_image_array = display_image_array

        self.app_process = Process(target=self._app_run)
        self.app_process.daemon = True # 设置为守护线程
        self.app_process.start()

    def _app_run(self) -> None:
        app = Vuer(host="0.0.0.0",
                   queries=dict(reconnect=True, grid=False),
                   queue_len=3)
        
        app.add_handler("HAND_MOVE")(self._on_hand_move)
        app.spawn(start=False)(self._app_main)

        app.run()

    async def _on_hand_move(self, event: ClientEvent, session: VuerSession) -> None:
        try:
            left = np.array(event.value["left"]).reshape(25, 16)
            self.left_hand_array[:] = left[0] # [16,]
            self.left_landmarks_array[:] = left[:, 12:15] # [25, 3]

            right = np.array(event.value["right"]).reshape(25, 16)
            self.right_hand_array[:] = right[0]
            self.right_landmarks_array[:] = right[:, 12:15]

            self.is_left_hand_initialized.value = True
            self.is_right_hand_initialized.value = True
            self.is_left_landmarks_initialized.value = True
            self.is_right_landmarks_initialized.value = True
        except:
            pass

    async def _app_main(self, session: VuerSession) -> None:
        session.upsert @ Hands(stream=True, key="hands")

        while True:
            session.upsert(ImageBackground(self.display_image_array,
                                           format="jpeg",
                                           quality=80,
                                           key="background",
                                           interpolate=True,
                                           fixed=True,
                                           distanceToCamera=1,
                                           position=[0, 0, -3],), to="bgChildren")

            await sleep(0.03)

    @property
    def left_hand_pose_matrix(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.is_left_hand_initialized:
                return self.left_hand_array.copy().reshape(4, 4, order='F')
            else:
                return None
        
    @property
    def right_hand_pose_matrix(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.is_right_hand_initialized:
                return self.right_hand_array.copy().reshape(4, 4, order='F')
            else:
                return None
    
    @property
    def left_landmarks_position(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.is_left_landmarks_initialized:
                return self.left_landmarks_array.copy()
            else:
                return None
        
    @property
    def right_landmarks_position(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.is_right_landmarks_initialized:
                return self.right_landmarks_array.copy()
            else:
                return None
            
    @property
    def intialized(self) -> Optional[np.ndarray]:
        return (self.is_left_hand_initialized.value and 
                self.is_right_hand_initialized.value and 
                self.is_left_landmarks_initialized.value and 
                self.is_right_landmarks_initialized.value)