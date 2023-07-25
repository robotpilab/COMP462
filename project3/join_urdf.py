import pybullet as p
from pybullet_utils import bullet_client as bc
from pybullet_utils import urdfEditor as ed
import pybullet_data
import time


if __name__ == "__main__":

    p0 = bc.BulletClient(connection_mode=p.DIRECT)
    p0.setAdditionalSearchPath(pybullet_data.getDataPath())

    p1 = bc.BulletClient(connection_mode=p.DIRECT)
    p1.setAdditionalSearchPath(pybullet_data.getDataPath())

    panda = p1.loadURDF("franka_panda/panda.urdf", flags=p1.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    stick = p0.loadURDF("urdf/stick.urdf")

    ed0 = ed.UrdfEditor()
    ed0.initializeFromBulletBody(panda, p1._client)
    ed1 = ed.UrdfEditor()
    ed1.initializeFromBulletBody(stick, p0._client)

    parentLinkIndex = 12

    jointPivotXYZInParent = [0.1, 0, 0]
    jointPivotRPYInParent = [0, 1.57, 0]

    jointPivotXYZInChild = [0, 0, 0]
    jointPivotRPYInChild = [0, 0, 0]

    newjoint = ed0.joinUrdf(ed1, parentLinkIndex, jointPivotXYZInParent, jointPivotRPYInParent,
                            jointPivotXYZInChild, jointPivotRPYInChild, p0._client, p1._client)
    newjoint.joint_type = p0.JOINT_FIXED

    ed0.saveUrdf("urdf/panda_stick.urdf")

    print(p0._client)
    print(p1._client)
    print("p0.getNumBodies()=", p0.getNumBodies())
    print("p1.getNumBodies()=", p1.getNumBodies())

    pgui = bc.BulletClient(connection_mode=p.GUI)
    pgui.configureDebugVisualizer(pgui.COV_ENABLE_RENDERING, 0)

    orn = [0, 0, 0, 1]
    ed0.createMultiBody([0, 0, 0], orn, pgui._client)
    pgui.setRealTimeSimulation(1)

    pgui.configureDebugVisualizer(pgui.COV_ENABLE_RENDERING, 1)

    st = time.time()
    while (pgui.isConnected()) and (time.time() - st < 2.0):
        pgui.getCameraImage(320, 200, renderer=pgui.ER_BULLET_HARDWARE_OPENGL)
        time.sleep(1. / 240.)