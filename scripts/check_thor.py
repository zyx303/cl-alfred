from ai2thor.controller import Controller
import debugpy
debugpy.listen(("localhost", 5678))
debugpy.wait_for_client()
c = Controller()
c.start()
event = c.step(dict(action="MoveAhead"))
assert event.frame.shape == (300, 300, 3)
print(event.frame.shape)
print("Everything works!!!")