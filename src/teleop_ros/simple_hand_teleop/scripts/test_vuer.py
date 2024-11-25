from asyncio import sleep

import imageio as iio
from tqdm import tqdm

from vuer import Vuer, VuerSession
from vuer.events import ClientEvent
from vuer.schemas import Scene, ImageBackground

import numpy as np

reader = iio.get_reader("/home/yilin/Downloads/sample-30s.webm")

app = Vuer()

# @app.add_handler("CAMERA_MOVE")
# async def on_camera(event: ClientEvent, sess: VuerSession):
#     assert event == "CAMERA_MOVE", "the event type should be correct"
#     print("camera event", event.etype, event.value)

@app.spawn(start=True)
async def show_heatmap(sess: VuerSession):
    sess.set @ Scene()

    for i, frame in tqdm(enumerate(reader), desc="playing video"):
        print(frame.shape, type(frame))
        print(frame[::2, ::2, :].shape, type(frame[::2, ::2, :]))
        frame_np = np.array(frame)
        print(frame_np.shape, type(frame_np))
        # use the upsert(..., to="bgChildren") syntax, so it is in global frame.
        sess.upsert(
            ImageBackground(

                # Can scale the images down.
                frame_np[::2, ::2, :],

                # One of ['b64png', 'png', 'b64jpeg', 'jpeg']
                # 'b64png' does not work for some reason, but works for the nerf demo.
                # 'jpeg' encoding is significantly faster than 'png'.
                format="jpeg",
                quality=20,
                key="background",
                interpolate=True,
                fixed=True,
                distanceToCamera=1,

                # can test with matrix
                # matrix=[
                #     1.2418025750411799, 0, 0, 0,
                #     0, 1.5346539759579207, 0, 0,
                #     0, 0, 1, 0,
                #     0, 0, -3, 1,
                # ],
                position=[0, 0, -3],
                ### Can also rotate the plane in-place.
                # rotation=[-0.25, 0, 0],
            ),
            # we place this into the background children list, so that it is
            # not affected by the global rotation
            to="bgChildren",
        )

        # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
        # this is mostly limited by the python server side.
        await sleep(0.016)