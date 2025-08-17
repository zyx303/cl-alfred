from ai2thor.controller import Controller
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()

import os
os.environ['ALFRED_ROOT'] = '/home/yongxi/work/cl-alfred'
os.environ['display']='0'


controller = Controller()
event = controller.step("MoveAhead")