# Configure the state settings below according to your own setup

RUN_WITH_CAMERA: bool = False
if(RUN_WITH_CAMERA):
  import camera.camera_module as camera_generic
  import camera.amscope.amscope_camera as amscope_camera
  import camera.flir.flir_camera as flir 
  camera = flir.FlirCamera() # instantiate an object implementing camera_module