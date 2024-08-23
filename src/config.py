# Configure the state settings below according to your own setup

RUN_WITH_CAMERA: bool = False
if(RUN_WITH_CAMERA):
  import camera.camera_module as camera_generic
  import camera.amscope.amscope_camera as amscope_camera
  import camera.flir.flir_camera as flir 
  camera = flir.FlirCamera() # instantiate an object implementing camera_module
  
RUN_WITH_STAGE: bool = False
if(RUN_WITH_STAGE):
  stage_file = "COM6" # change as needed; later will have automatic COM port identification
  baud_rate = 115200
  scale_factor = 50/3890 # For us, 50 units went about 3890 microns. PLEASE CHANGE THIS FOR YOUR OWN SETUP.
  
