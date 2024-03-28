import os
import logging
import vtk
import re
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import sitkUtils as su
import SimpleITK as sitk
import vtkSegmentationCorePython as vtkSegmentationCore
import torch
import sysconfig
try:
   from tumorsegmentator.python_api import tumorsegmentator
except:
  slicer.util.pip_install('git+https://github.com/ReubenDo/TumorSegmentator#egg=TumorSegmentator')
  # slicer.util.pip_install('tumorsegmentator')
  from tumorsegmentator.python_api import tumorsegmentator

try:
   from HD_BET.run import run_hd_bet
except:
  slicer.util.pip_install('git+https://github.com/ReubenDo/HD-BET#egg=HD-BET')
  # slicer.util.pip_install('tumorsegmentator')
  from HD_BET.run import run_hd_bet
  
from pathlib import Path
#
# SlicerTumorSegmentator
#
REPO = 'https://github.com/ReubenDo/SlicerSlicerTumorSegmentator/'

PARAMETERS_BRAINSFIT = ['--samplingPercentage', '0.02',
                        '--useRigid', '--useAffine',
                        '--initializeTransformMode', 'useCenterOfHeadAlign' ]


class SlicerTumorSegmentator(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = 'Brain Tumor Segmentator'
    self.parent.categories = ['Segmentation']
    self.parent.dependencies = []
    self.parent.contributors = ["Reuben Dorent (Harvard University)"]  
    self.parent.helpText = (
      'Brain Tumor segmentator based on SlicerTotalSegmentator.'
      f'<p>Code: <a href="{REPO}">here</a>.</p>'
    )
    # self.parent.acknowledgementText = (
    #   'This work was was funded by the Engineering and Physical Sciences'
    #   ' Research Council (EPSRC) and supported by the School of Biomedical Engineering'
    #   " & Imaging Sciences (BMEIS) of King's College London."
    # )

#
# SlicerTumorSegmentatorWidget
#
class SlicerTumorSegmentatorWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/SlicerTumorSegmentator.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = SlicerTumorSegmentatorLogic()
    self.logic.logCallback = self.addLog
    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.cet1Selector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.t2Selector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.flairSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.refSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    # Buttons
    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()
    
  def addLog(self, text):
      """Append text to log window
      """
      self.ui.statusLabel.appendPlainText(text)
      slicer.app.processEvents()  # force update
    

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    if not self._parameterNode.GetNodeReference("InputVolumet1"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolumet1", firstVolumeNode.GetID())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    # Update node selectors and sliders
    self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolumet1"))
    self.ui.cet1Selector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolumecet1"))
    self.ui.t2Selector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolumet2"))
    self.ui.flairSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolumeflair"))
    self.ui.refSelector.setCurrentNode(self._parameterNode.GetNodeReference("RefVolume"))
    self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
    self.ui.checkCoregistered.checked = self._parameterNode.GetParameter("Coregistered") == "true"
    self.ui.checkSkullstripped.checked = self._parameterNode.GetParameter("Skullstripped") == "true"
    self.ui.checkDeleteProcessed.checked = self._parameterNode.GetParameter("DeleteProcessed") == "true"
    
    #self.ui.progressBar1.setCurrentNode(self._parameterNode.GetNodeReference("ProgressBar1"))

    # Update buttons states and tooltips
    if self._parameterNode.GetNodeReference("RefVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
      self.ui.applyButton.toolTip = "Compute output volume"
      self.ui.applyButton.enabled = True
    else:
      self.ui.applyButton.toolTip = "Select input and output volume nodes"
      self.ui.applyButton.enabled = False

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    self._parameterNode.SetNodeReferenceID("InputVolumet1", self.ui.inputSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolumecet1", self.ui.cet1Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolumet2", self.ui.t2Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolumeflair", self.ui.flairSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("RefVolume", self.ui.refSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
    
    self._parameterNode.SetParameter("Coregistered", "true" if self.ui.checkCoregistered.checked else "false")
    self._parameterNode.SetParameter("Skullstripped", "true" if self.ui.checkSkullstripped.checked else "false")
    self._parameterNode.SetParameter("DeletePreprocessed", "true" if self.ui.checkDeleteProcessed.checked else "false")

    self._parameterNode.EndModify(wasModified)

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    try:
      # Compute output
      self.logic.process(
        self.ui.refSelector.currentNode(),
        self.ui.inputSelector.currentNode(), 
        self.ui.cet1Selector.currentNode(), 
        self.ui.t2Selector.currentNode(), 
        self.ui.flairSelector.currentNode(), 
        self.ui.outputSelector.currentNode(),
        self.ui.checkCoregistered.checked, 
        self.ui.checkSkullstripped.checked,
        self.ui.checkDeleteProcessed.checked)

    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()


#
# SlicerTumorSegmentatorLogic
#

class SlicerTumorSegmentatorLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)
    self.logCallback = None
    

  def log(self, text):
      logging.info(text)
      if self.logCallback:
          self.logCallback(text)
  
  def logProcessOutput(self, proc, returnOutput=False):
      # Wait for the process to end and forward output to the log
      output = ""
      from subprocess import CalledProcessError
      while True:
          try:
              line = proc.stdout.readline()
              if not line:
                  break
              if returnOutput:
                  output += line
              self.log(line.rstrip())
          except UnicodeDecodeError as e:
              # Code page conversion happens because `universal_newlines=True` sets process output to text mode,
              # and it fails because probably system locale is not UTF8. We just ignore the error and discard the string,
              # as we only guarantee correct behavior if an UTF8 locale is used.
              pass

      proc.wait()
      retcode = proc.returncode
      if retcode != 0:
          raise CalledProcessError(retcode, proc.args, output=proc.stdout, stderr=proc.stderr)
      return output if returnOutput else None

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    pass
  
  def readSegmentationFolder(self, outputSegmentation, output_segmentation_dir, task="tumor"):
      """
      The method is very slow, but this is the only option for some specialized tasks.
      """

      import os

      outputSegmentation.GetSegmentation().RemoveAllSegments()

      # Get color node with random colors
      randomColorsNode = slicer.mrmlScene.GetNodeByID('vtkMRMLColorTableNodeRandom')
      rgba = [0, 0, 0, 0]
      # Get label descriptions

      # Get label descriptions if task is provided
      from tumorsegmentator.map_to_binary import class_map
      labelValueToSegmentName = class_map[task] if task else {}

      def import_labelmap_to_segmentation(labelmapVolumeNode, segmentId):
          updatedSegmentIds = vtk.vtkStringArray()
          updatedSegmentIds.InsertNextValue(segmentId)
          slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, outputSegmentation, updatedSegmentIds)
          slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

      # Read each candidate file
      for labelValue, segmentName in labelValueToSegmentName.items():
          self.log(f"Importing candidate {segmentName}")
          candidate = [k for k in os.listdir(output_segmentation_dir) if segmentName in k][0]
          labelVolumePath = os.path.join(output_segmentation_dir, candidate)
          if not os.path.exists(labelVolumePath):
              self.log(f"Path {segmentName} not exists.")
              continue
          labelmapVolumeNode = slicer.util.loadLabelVolume(labelVolumePath, {"name": segmentName})
          randomColorsNode.GetColor(labelValue, rgba)
          segmentId = outputSegmentation.GetSegmentation().AddEmptySegment(segmentName, segmentName, rgba[0:3])
          import_labelmap_to_segmentation(labelmapVolumeNode, segmentId)

          
  def readSegmentation(self, outputSegmentation, outputSegmentationFile, task='tumor'):

    # Get label descriptions
    from tumorsegmentator.map_to_binary import class_map
    labelValueToSegmentName = class_map[task]
    maxLabelValue = max(labelValueToSegmentName.keys())
    if min(labelValueToSegmentName.keys()) < 0:
        raise RuntimeError("Label values in class_map must be positive")

    # Get color node with random colors
    randomColorsNode = slicer.mrmlScene.GetNodeByID('vtkMRMLColorTableNodeRandom')
    rgba = [0, 0, 0, 0]

    # Create color table for this segmentation task
    colorTableNode = slicer.vtkMRMLColorTableNode()
    colorTableNode.SetTypeToUser()
    colorTableNode.SetNumberOfColors(maxLabelValue+1)
    colorTableNode.SetName(task)
    for labelValue in labelValueToSegmentName:
        randomColorsNode.GetColor(labelValue,rgba)
        colorTableNode.SetColor(labelValue, rgba[0], rgba[1], rgba[2], rgba[3])
        colorTableNode.SetColorName(labelValue, labelValueToSegmentName[labelValue])
    slicer.mrmlScene.AddNode(colorTableNode)

    # Load the segmentation
    # outputSegmentation.SetLabelmapConversionColorTableNodeID(colorTableNode.GetID())
    outputSegmentation.AddDefaultStorageNode()
    storageNode = outputSegmentation.GetStorageNode()
    storageNode.SetFileName(outputSegmentationFile)
    storageNode.ReadData(outputSegmentation)

    slicer.mrmlScene.RemoveNode(colorTableNode)
    
  def _apply_mask(self, input, output, mask):
      # Reorient in case HD-BET changed the orientation of the raw file
      input_img = sitk.ReadImage(input)
      mask = sitk.ReadImage(mask)
      masked_image = sitk.Mask(input_img, mask==1, outsideValue=0)
      sitk.WriteImage(masked_image, output)


  def process(self, 
              refVolume, 
              inputVolumet1, 
              inputVolumecet1, 
              inputVolumet2, 
              inputVolumeflair, 
              outputVolume, 
              coregistered, 
              skullstripped,
              deleteprocessed):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be thresholded
    :param outputVolume: thresholding result
    """

    if not inputVolumet1 or not outputVolume:
      raise ValueError("Input or output volume is invalid")
    
    import time
    startTime = time.time()
    self.log('Processing started')
    
    
    if torch.backends.cuda.is_built() and torch.cuda.is_available():
      device = "cuda"
    else:
      device = "cpu"
    tempFolder = slicer.util.tempDirectory()
    
    import shutil
    pythonSlicerExecutablePath = shutil.which('PythonSlicer')
    if not pythonSlicerExecutablePath:
        raise RuntimeError("Python was not found")
      
    refFile = tempFolder+"/tumor-segmentator-ref.nii.gz"
    inputFilet1 = tempFolder+"/tumor-segmentator-t1.nii.gz"
    inputFilecet1 = tempFolder+"/tumor-segmentator-cet1.nii.gz"
    inputFilet2 = tempFolder+"/tumor-segmentator-t2.nii.gz"
    inputFileflair = tempFolder+"/tumor-segmentator-flair.nii.gz"
  
    
    # Write input volume to file
    # TumorSegmentator requires NIFTI
    self.log(f"Writing input file to {refFile}")
    volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
    volumeStorageNode.SetFileName(refFile)
    volumeStorageNode.UseCompressionOff()
    volumeStorageNode.WriteData(refVolume)
    volumeStorageNode.UnRegister(None)
    
    self.log(f"Writing input file to {inputFilet1}")
    volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
    volumeStorageNode.SetFileName(inputFilet1)
    volumeStorageNode.UseCompressionOff()
    volumeStorageNode.WriteData(inputVolumet1)
    volumeStorageNode.UnRegister(None)
    
    self.log(f"Writing input file to {inputFilecet1}")
    volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
    volumeStorageNode.SetFileName(inputFilecet1)
    volumeStorageNode.UseCompressionOff()
    volumeStorageNode.WriteData(inputVolumecet1)
    volumeStorageNode.UnRegister(None)
    
    options = ["-t1", inputFilet1, "-cet1", inputFilecet1, "-o", tempFolder]
    if not inputVolumet2 is None and not inputVolumeflair is None:
    
      self.log(f"Writing input file to {inputFilet2}")
      volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
      volumeStorageNode.SetFileName(inputFilet2)
      volumeStorageNode.UseCompressionOff()
      volumeStorageNode.WriteData(inputVolumet2)
      volumeStorageNode.UnRegister(None)
      options.append('-t2')
      options.append(inputFilet2)

      self.log(f"Writing input file to {inputFileflair}")
      volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
      volumeStorageNode.SetFileName(inputFileflair)
      volumeStorageNode.UseCompressionOff()
      volumeStorageNode.WriteData(inputVolumeflair)
      volumeStorageNode.UnRegister(None)
      options.append('-flair')
      options.append(inputFileflair)  
      
    if not coregistered:
      self.log(f"Performing co-registration")
      list_files = [k for k in os.listdir(tempFolder) if 'nii' in k and not 'ref' in k]
      if refVolume.GetID()==inputVolumet1.GetID():
        list_files = [k for k in list_files if not '-t1.' in k]
      elif refVolume.GetID()==inputVolumecet1.GetID():
        list_files = [k for k in list_files if not '-cet1.' in k]
      elif refVolume.GetID()==inputVolumet2.GetID():
        list_files = [k for k in list_files if not '-t2.' in k]
      elif refVolume.GetID()==inputVolumeflair.GetID():
        list_files = [k for k in list_files if not '-flair.' in k]
        
      for file in list_files:
        
        inputFileTemp = os.path.join(tempFolder, file)
        self.log(f"Performing co-registered input file to {inputFileTemp}")
        BRAINSFITExecutablePath = shutil.which('BRAINSFit')
        options_brainsfit = ['--fixedVolume', refFile, 
                             '--movingVolume', inputFileTemp, 
                             '--outputVolume', inputFileTemp]  
        self.log(' '.join([BRAINSFITExecutablePath] + options_brainsfit))    
        proc = slicer.util.launchConsoleProcess([BRAINSFITExecutablePath] + options_brainsfit + PARAMETERS_BRAINSFIT)
        self.logProcessOutput(proc)
        
        
    if not skullstripped:
      self.log(f"Performing skullstripping")
      mask_sk = tempFolder+"/tumor-segmentator-ref_masked_mask.nii.gz"
      output_sk = tempFolder+"/tumor-segmentator-ref_masked.nii.gz"
      HDBETExecutablePath = os.path.join(sysconfig.get_path('scripts'), "hd-bet")
      HDBETCommand = [ pythonSlicerExecutablePath, HDBETExecutablePath]
      options_hdbet = ['-i', refFile, '-o', output_sk, '-device', device, '-mode', 'fast', '-tta', '0']
      self.log(f"Tumor Segmentator arguments: {HDBETCommand + options_hdbet}")
      proc = slicer.util.launchConsoleProcess(HDBETCommand + options_hdbet)
      self.logProcessOutput(proc)
      
      # run_hd_bet(refFile, refFile, mode="fast", device=device, do_tta=False, )
      list_files = [k for k in os.listdir(tempFolder) if 'nii' in k and not 'ref' in k]
      for file in list_files:
        inputFileTemp = os.path.join(tempFolder, file)
        self._apply_mask(input=inputFileTemp, output=inputFileTemp, mask=mask_sk)
        
    if not deleteprocessed:
      list_files = [os.path.join(tempFolder,k) for k in os.listdir(tempFolder) if 'nii' in k and not 'ref' in k]
      for file in list_files:
        slicer.util.loadVolume(file, returnNode=False)
        
    # Recommend the user to switch to fast mode if no GPU or not enough memory is available    
    options.extend(["--device", device])

    # Get TumorSegmentator launcher command
    # TumorSegmentator (.py file, without extension) is installed in Python Scripts folder
    
    TumorSegmentatorExecutablePath = os.path.join(sysconfig.get_path('scripts'), "TumorSegmentator")
    # TumorSegmentatorExecutablePath = "TumorSegmentator"
    # Get Python executable path
    TumorSegmentatorCommand = [ pythonSlicerExecutablePath, TumorSegmentatorExecutablePath]

    # Get options
    self.log('Creating segmentations with TumorSegmentator...')
    self.log(f"Tumor Segmentator arguments: {options}")
    proc = slicer.util.launchConsoleProcess(TumorSegmentatorCommand + options)
    self.logProcessOutput(proc)

    # Load result
    self.log('Importing segmentation results...')

    # Create output labelmap
    self.readSegmentationFolder(outputVolume, tempFolder)
    
    self.log("Cleaning up temporary folder...")
    if os.path.isdir(tempFolder):
        shutil.rmtree(tempFolder)
    
    # Set source volume - required for DICOM Segmentation export
    outputVolume.SetNodeReferenceID(outputVolume.GetReferenceImageGeometryReferenceRole(), refVolume.GetID())
    outputVolume.SetReferenceImageGeometryParameterFromVolumeNode(refVolume)

    # Place segmentation node in the same place as the input volume
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    inputVolumeShItem = shNode.GetItemByDataNode(refVolume)
    studyShItem = shNode.GetItemParent(inputVolumeShItem)
    segmentationShItem = shNode.GetItemByDataNode(outputVolume)
    shNode.SetItemParent(segmentationShItem, studyShItem)

    stopTime = time.time()
    self.log(f'Processing completed in {stopTime-startTime:.2f} seconds')

#
# SlicerTumorSegmentatorTest
#

class SlicerTumorSegmentatorTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_SlicerSlicerTumorSegmentator1()

  def test_SlicerSlicerTumorSegmentator1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    # Get/create input data

    import SampleData
    # registerSampleData()
    inputVolume = SampleData.downloadSample('MRBrainTumor1')
    self.delayDisplay('Loaded test data set')

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

    # Test the module logic

    logic = SlicerTumorSegmentatorLogic()

    # Test algorithm 
    logic.process(inputVolume, outputVolume)

    self.delayDisplay('Test passed')
