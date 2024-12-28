import statistics as stats
import numpy as np
from math import isnan

CIFAR10_ResNet_SGD_Trials = []
CIFAR10_ResNet_SGD_Trials = []
CIFAR10_ResNet_SGD_Trials = []
CIFAR10_ResNet_SGD_Trials = []

CIFAR10_ResNet_SGD_Trials = []
CIFAR10_ResNet_SGD_Trials = []
CIFAR10_ResNet_SGD_Trials = []
CIFAR10_ResNet_SGD_Trials = []

CIFAR10_WideResNet_SGD_Trials = []
CIFAR10_WideResNet_SGD_Trials = []
CIFAR10_WideResNet_SGD_Trials = []
CIFAR10_WideResNet_SGD_Trials = []

#Read ResNet SGD Trials
for i in range(1,6):
  tempFile = open('ResNet50/SGD/ResNet_CIFAR10_Trial'+str(i)+'.csv', 'r')
  tempTrial = []

  #Read lines
  SishTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  SmishTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  LogishTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  MishTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  SwishTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  SishValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  SmishValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  LogishValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  MishValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  SwishValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  SishTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  SmishTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  LogishTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  MishTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  SwishTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  SishValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  SmishValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  LogishValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  MishValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  SwishValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  SishTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  SmishTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  LogishTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  MishTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  SwishTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  SishValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  SmishValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  LogishValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  MishValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  SwishValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  SishTestingAccuracy = float(tempFile.readline().replace("\"",""))
  GELUTestingAccuracy = float(tempFile.readline().replace("\"",""))
  SmishTestingAccuracy = float(tempFile.readline().replace("\"",""))
  LogishTestingAccuracy = float(tempFile.readline().replace("\"",""))
  ReLUTestingAccuracy = float(tempFile.readline().replace("\"",""))
  MishTestingAccuracy = float(tempFile.readline().replace("\"",""))
  SwishTestingAccuracy = float(tempFile.readline().replace("\"",""))

  #append CSV lines to temp trial list of metrics
  tempTrial.append(SishTrainingAccuracy)
  tempTrial.append(GELUTrainingAccuracy)
  tempTrial.append(SmishTrainingAccuracy)
  tempTrial.append(LogishTrainingAccuracy)
  tempTrial.append(ReLUTrainingAccuracy)
  tempTrial.append(MishTrainingAccuracy)
  tempTrial.append(SwishTrainingAccuracy)

  tempTrial.append(SishValidationAccuracy)
  tempTrial.append(GELUValidationAccuracy)
  tempTrial.append(SmishValidationAccuracy)
  tempTrial.append(LogishValidationAccuracy)
  tempTrial.append(ReLUValidationAccuracy)
  tempTrial.append(MishValidationAccuracy)
  tempTrial.append(SwishValidationAccuracy)

  tempTrial.append(SishTrainingError)
  tempTrial.append(GELUTrainingError)
  tempTrial.append(SmishTrainingError)
  tempTrial.append(LogishTrainingError)
  tempTrial.append(ReLUTrainingError)
  tempTrial.append(MishTrainingError)
  tempTrial.append(SwishTrainingError)

  tempTrial.append(SishValidationError)
  tempTrial.append(GELUValidationError)
  tempTrial.append(SmishValidationError)
  tempTrial.append(LogishValidationError)
  tempTrial.append(ReLUValidationError)
  tempTrial.append(MishValidationError)
  tempTrial.append(SwishValidationError)

  tempTrial.append(SishTrainingRuntime)
  tempTrial.append(GELUTrainingRuntime)
  tempTrial.append(SmishTrainingRuntime)
  tempTrial.append(LogishTrainingRuntime)
  tempTrial.append(ReLUTrainingRuntime)
  tempTrial.append(MishTrainingRuntime)
  tempTrial.append(SwishTrainingRuntime)

  tempTrial.append(SishValidationRuntime)
  tempTrial.append(GELUValidationRuntime)
  tempTrial.append(SmishValidationRuntime)
  tempTrial.append(LogishValidationRuntime)
  tempTrial.append(ReLUValidationRuntime)
  tempTrial.append(MishValidationRuntime)
  tempTrial.append(SwishValidationRuntime)

  tempTrial.append(SishTestingAccuracy)
  tempTrial.append(GELUTestingAccuracy)
  tempTrial.append(SmishTestingAccuracy)
  tempTrial.append(LogishTestingAccuracy)
  tempTrial.append(ReLUTestingAccuracy)
  tempTrial.append(MishTestingAccuracy)
  tempTrial.append(SwishTestingAccuracy)

  #append trial metric lists to trials lists
  CIFAR10_ResNet_SGD_Trials.append(tempTrial)
  tempFile.close()

#Prepare data from not-a-number values #REWORK??? todo: todo
for i in range(len(CIFAR10_ResNet_SGD_Trials)):
  for j in range(len(CIFAR10_ResNet_SGD_Trials[i][14])):
    if isnan(CIFAR10_ResNet_SGD_Trials[i][14][j]):
      CIFAR10_ResNet_SGD_Trials[i][14][j] = 1000.
    if isnan(CIFAR10_ResNet_SGD_Trials[i][15][j]):
      CIFAR10_ResNet_SGD_Trials[i][15][j] = 1000.
    if isnan(CIFAR10_ResNet_SGD_Trials[i][16][j]):
      CIFAR10_ResNet_SGD_Trials[i][16][j] = 1000.
    if isnan(CIFAR10_ResNet_SGD_Trials[i][17][j]):
      CIFAR10_ResNet_SGD_Trials[i][17][j] = 1000.
    if isnan(CIFAR10_ResNet_SGD_Trials[i][18][j]):
      CIFAR10_ResNet_SGD_Trials[i][18][j] = 1000.
    if isnan(CIFAR10_ResNet_SGD_Trials[i][19][j]):
      CIFAR10_ResNet_SGD_Trials[i][19][j] = 1000.
    if isnan(CIFAR10_ResNet_SGD_Trials[i][20][j]):
      CIFAR10_ResNet_SGD_Trials[i][20][j] = 1000.

print("ResNet SGD:")
print()

#Show Sish metrics for SGD on ResNet:
# for i in range(5):
#   print(CIFAR10_ResNet_SGD_Trials[i][43])
SishTestingAccuracy = stats.mean([CIFAR10_ResNet_SGD_Trials[i][42] for i in range(5)])
SishConverganceTestingStdv = stats.stdev([CIFAR10_ResNet_SGD_Trials[i][42] for i in range(5)])

SishConverganceValidationAccuracy = stats.mean([np.max(CIFAR10_ResNet_SGD_Trials[i][7]) for i in range(5)])
SishConverganceValidationMedian = stats.median([np.max(CIFAR10_ResNet_SGD_Trials[i][7]) for i in range(5)])
SishConverganceValidationStdv = stats.stdev([np.max(CIFAR10_ResNet_SGD_Trials[i][7]) for i in range(5)])

SishConverganceEpochIndeces = [np.argmax(CIFAR10_ResNet_SGD_Trials[i][7]) for i in range(5)]
SishConverganceEpochFloats = [float(SishConverganceEpochIndex) for SishConverganceEpochIndex in SishConverganceEpochIndeces]
SishConverganceEpochAverage = stats.mean(SishConverganceEpochFloats)

SishConverganceTrainingAccuracy = stats.mean([CIFAR10_ResNet_SGD_Trials[i][0][SishConverganceEpochIndeces[i]] for i in range(5)])

SishFittingIndex = (100-SishConverganceTrainingAccuracy)/(100-SishTestingAccuracy)

SishMinTrainingError = stats.mean([np.min(CIFAR10_ResNet_SGD_Trials[i][14]) for i in range(5)])

SishMeanTrainingTime = stats.mean([stats.mean(CIFAR10_ResNet_SGD_Trials[i][28]) for i in range(5)])

SishMeanValidationTime = stats.mean([stats.mean(CIFAR10_ResNet_SGD_Trials[i][35]) for i in range(5)])

print(f"Average Sish convergance Training accuracy: {SishConverganceTrainingAccuracy:.3f}")
print(f"Average Sish convergance Validation accuracy: {SishConverganceValidationAccuracy:.3f}")
print(f"Average Sish convergance Testing accuracy: {SishTestingAccuracy:.3f}")
#print(f"Median Sish convergance Testing accuracy: {SishConverganceTestingMedian:.3f}")
#print(f"Stdv of Sish convergance Validation accuracy: {SishConverganceValidationStdv:.3f}")
print(f"Average Sish convergance epoch: {SishConverganceEpochAverage:.3f}")
print(f"Average Sish fitting index: {SishFittingIndex:.3f}")
print(f"Stdv of Sish convergance test accuracy: {SishConverganceTestingStdv:.3f}")
#print(f"Average Sish minimum training error: {SishMinTrainingError:.3f}")
#print(f"Average Sish epoch runtime: {SishMeanTrainingTime:.3f}")
#print(f"Average Sish testing runtime: {SishMeanTestingTime:.3f}")
print()

#Show GELU metrics for SGD on ResNet:
GELUTestingAccuracy = stats.mean([CIFAR10_ResNet_SGD_Trials[i][43] for i in range(5)])
GELUConverganceTestingStdv = stats.stdev([CIFAR10_ResNet_SGD_Trials[i][43] for i in range(5)])

GELUConverganceValidationAccuracy = stats.mean([np.max(CIFAR10_ResNet_SGD_Trials[i][8]) for i in range(5)])
GELUConverganceValidationMedian = stats.median([np.max(CIFAR10_ResNet_SGD_Trials[i][8]) for i in range(5)])
GELUConverganceValidationStdv = stats.stdev([np.max(CIFAR10_ResNet_SGD_Trials[i][8]) for i in range(5)])

GELUConverganceEpochIndeces = [np.argmax(CIFAR10_ResNet_SGD_Trials[i][8]) for i in range(5)]
GELUConverganceEpochFloats = [float(GELUConverganceEpochIndex) for GELUConverganceEpochIndex in GELUConverganceEpochIndeces]
GELUConverganceEpochAverage = stats.mean(GELUConverganceEpochFloats)

GELUConverganceTrainingAccuracy = stats.mean([CIFAR10_ResNet_SGD_Trials[i][1][GELUConverganceEpochIndeces[i]] for i in range(5)])

GELUFittingIndex = (100-GELUConverganceTrainingAccuracy)/(100-GELUTestingAccuracy)

GELUMinTrainingError = stats.mean([np.min(CIFAR10_ResNet_SGD_Trials[i][15]) for i in range(5)])

GELUMeanTrainingTime = stats.mean([stats.mean(CIFAR10_ResNet_SGD_Trials[i][29]) for i in range(5)])

GELUMeanValidationTime = stats.mean([stats.mean(CIFAR10_ResNet_SGD_Trials[i][36]) for i in range(5)])

print(f"Average GELU convergance Training accuracy: {GELUConverganceTrainingAccuracy:.3f}")
print(f"Average GELU convergance Validation accuracy: {GELUConverganceValidationAccuracy:.3f}")
print(f"Average GELU convergance Testing accuracy: {GELUTestingAccuracy:.3f}")
#print(f"Median GELU convergance Testing accuracy: {GELUConverganceTestingMedian:.3f}")
#print(f"Stdv of GELU convergance Validation accuracy: {GELUConverganceValidationStdv:.3f}")
print(f"Average GELU convergance epoch: {GELUConverganceEpochAverage:.3f}")
print(f"Average GELU fitting index: {GELUFittingIndex:.3f}")
print(f"Stdv of GELU convergance test accuracy: {GELUConverganceTestingStdv:.3f}")
#print(f"Average GELU minimum training error: {GELUMinTrainingError:.3f}")
#print(f"Average GELU epoch runtime: {GELUMeanTrainingTime:.3f}")
#print(f"Average GELU testing runtime: {GELUMeanTestingTime:.3f}")
print()

#Show Smish metrics for SGD on ResNet:
SmishTestingAccuracy = stats.mean([CIFAR10_ResNet_SGD_Trials[i][44] for i in range(5)])
SmishConverganceTestingStdv = stats.stdev([CIFAR10_ResNet_SGD_Trials[i][44] for i in range(5)])

SmishConverganceValidationAccuracy = stats.mean([np.max(CIFAR10_ResNet_SGD_Trials[i][9]) for i in range(5)])
SmishConverganceValidationMedian = stats.median([np.max(CIFAR10_ResNet_SGD_Trials[i][9]) for i in range(5)])
SmishConverganceValidationStdv = stats.stdev([np.max(CIFAR10_ResNet_SGD_Trials[i][9]) for i in range(5)])

SmishConverganceEpochIndeces = [np.argmax(CIFAR10_ResNet_SGD_Trials[i][9]) for i in range(5)]
SmishConverganceEpochFloats = [float(SmishConverganceEpochIndex) for SmishConverganceEpochIndex in SmishConverganceEpochIndeces]
SmishConverganceEpochAverage = stats.mean(SmishConverganceEpochFloats)

SmishConverganceTrainingAccuracy = stats.mean([CIFAR10_ResNet_SGD_Trials[i][2][SmishConverganceEpochIndeces[i]] for i in range(5)])

SmishFittingIndex = (100-SmishConverganceTrainingAccuracy)/(100-SmishTestingAccuracy)

SmishMinTrainingError = stats.mean([np.min(CIFAR10_ResNet_SGD_Trials[i][16]) for i in range(5)])

SmishMeanTrainingTime = stats.mean([stats.mean(CIFAR10_ResNet_SGD_Trials[i][30]) for i in range(5)])

SmishMeanValidationTime = stats.mean([stats.mean(CIFAR10_ResNet_SGD_Trials[i][37]) for i in range(5)])

print(f"Average Smish convergance Training accuracy: {SmishConverganceTrainingAccuracy:.3f}")
print(f"Average Smish convergance Validation accuracy: {SmishConverganceValidationAccuracy:.3f}")
print(f"Average Smish convergance Testing accuracy: {SmishTestingAccuracy:.3f}")
#print(f"Median Smish convergance Testing accuracy: {SmishConverganceTestingMedian:.3f}")
#print(f"Stdv of Smish convergance Validation accuracy: {SmishConverganceValidationStdv:.3f}")
print(f"Average Smish convergance epoch: {SmishConverganceEpochAverage:.3f}")
print(f"Average Smish fitting index: {SmishFittingIndex:.3f}")
print(f"Stdv of Smish convergance test accuracy: {SmishConverganceTestingStdv:.3f}")
#print(f"Average Smish minimum training error: {SmishMinTrainingError:.3f}")
#print(f"Average Smish epoch runtime: {SmishMeanTrainingTime:.3f}")
#print(f"Average Smish testing runtime: {SmishMeanTestingTime:.3f}")
print()

#Show Logish metrics for SGD on ResNet:
LogishTestingAccuracy = stats.mean([CIFAR10_ResNet_SGD_Trials[i][45] for i in range(5)])
LogishConverganceTestingStdv = stats.stdev([CIFAR10_ResNet_SGD_Trials[i][45] for i in range(5)])

LogishConverganceValidationAccuracy = stats.mean([np.max(CIFAR10_ResNet_SGD_Trials[i][10]) for i in range(5)])
LogishConverganceValidationMedian = stats.median([np.max(CIFAR10_ResNet_SGD_Trials[i][10]) for i in range(5)])
LogishConverganceValidationStdv = stats.stdev([np.max(CIFAR10_ResNet_SGD_Trials[i][10]) for i in range(5)])

LogishConverganceEpochIndeces = [np.argmax(CIFAR10_ResNet_SGD_Trials[i][10]) for i in range(5)]
LogishConverganceEpochFloats = [float(LogishConverganceEpochIndex) for LogishConverganceEpochIndex in LogishConverganceEpochIndeces]
LogishConverganceEpochAverage = stats.mean(LogishConverganceEpochFloats)

LogishConverganceTrainingAccuracy = stats.mean([CIFAR10_ResNet_SGD_Trials[i][3][LogishConverganceEpochIndeces[i]] for i in range(5)])

LogishFittingIndex = (100-LogishConverganceTrainingAccuracy)/(100-LogishTestingAccuracy)

LogishMinTrainingError = stats.mean([np.min(CIFAR10_ResNet_SGD_Trials[i][17]) for i in range(5)])

LogishMeanTrainingTime = stats.mean([stats.mean(CIFAR10_ResNet_SGD_Trials[i][31]) for i in range(5)])

LogishMeanValidationTime = stats.mean([stats.mean(CIFAR10_ResNet_SGD_Trials[i][38]) for i in range(5)])

print(f"Average Logish convergance Training accuracy: {LogishConverganceTrainingAccuracy:.3f}")
print(f"Average Logish convergance Validation accuracy: {LogishConverganceValidationAccuracy:.3f}")
print(f"Average Logish convergance Testing accuracy: {LogishTestingAccuracy:.3f}")
#print(f"Median Logish convergance Testing accuracy: {LogishConverganceTestingMedian:.3f}")
#print(f"Stdv of Logish convergance Validation accuracy: {LogishConverganceValidationStdv:.3f}")
print(f"Average Logish convergance epoch: {LogishConverganceEpochAverage:.3f}")
print(f"Average Logish fitting index: {LogishFittingIndex:.3f}")
print(f"Stdv of Logish convergance test accuracy: {LogishConverganceTestingStdv:.3f}")
#print(f"Average Logish minimum training error: {LogishMinTrainingError:.3f}")
#print(f"Average Logish epoch runtime: {LogishMeanTrainingTime:.3f}")
#print(f"Average Logish testing runtime: {LogishMeanTestingTime:.3f}")
print()

#Show ReLU metrics for SGD on ResNet:
ReLUTestingAccuracy = stats.mean([CIFAR10_ResNet_SGD_Trials[i][46] for i in range(5)])
ReLUConverganceTestingStdv = stats.stdev([CIFAR10_ResNet_SGD_Trials[i][46] for i in range(5)])

ReLUConverganceValidationAccuracy = stats.mean([np.max(CIFAR10_ResNet_SGD_Trials[i][11]) for i in range(5)])
ReLUConverganceValidationMedian = stats.median([np.max(CIFAR10_ResNet_SGD_Trials[i][11]) for i in range(5)])
ReLUConverganceValidationStdv = stats.stdev([np.max(CIFAR10_ResNet_SGD_Trials[i][11]) for i in range(5)])

ReLUConverganceEpochIndeces = [np.argmax(CIFAR10_ResNet_SGD_Trials[i][11]) for i in range(5)]
ReLUConverganceEpochFloats = [float(ReLUConverganceEpochIndex) for ReLUConverganceEpochIndex in ReLUConverganceEpochIndeces]
ReLUConverganceEpochAverage = stats.mean(ReLUConverganceEpochFloats)

ReLUConverganceTrainingAccuracy = stats.mean([CIFAR10_ResNet_SGD_Trials[i][4][ReLUConverganceEpochIndeces[i]] for i in range(5)])

ReLUFittingIndex = (100-ReLUConverganceTrainingAccuracy)/(100-ReLUTestingAccuracy)

ReLUMinTrainingError = stats.mean([np.min(CIFAR10_ResNet_SGD_Trials[i][18]) for i in range(5)])

ReLUMeanTrainingTime = stats.mean([stats.mean(CIFAR10_ResNet_SGD_Trials[i][32]) for i in range(5)])

ReLUMeanValidationTime = stats.mean([stats.mean(CIFAR10_ResNet_SGD_Trials[i][39]) for i in range(5)])

print(f"Average ReLU convergance Training accuracy: {ReLUConverganceTrainingAccuracy:.3f}")
print(f"Average ReLU convergance Validation accuracy: {ReLUConverganceValidationAccuracy:.3f}")
print(f"Average ReLU convergance Testing accuracy: {ReLUTestingAccuracy:.3f}")
#print(f"Median ReLU convergance Testing accuracy: {ReLUConverganceTestingMedian:.3f}")
#print(f"Stdv of ReLU convergance Validation accuracy: {ReLUConverganceValidationStdv:.3f}")
print(f"Average ReLU convergance epoch: {ReLUConverganceEpochAverage:.3f}")
print(f"Average ReLU fitting index: {ReLUFittingIndex:.3f}")
print(f"Stdv of ReLU convergance test accuracy: {ReLUConverganceTestingStdv:.3f}")
#print(f"Average ReLU minimum training error: {ReLUMinTrainingError:.3f}")
#print(f"Average ReLU epoch runtime: {ReLUMeanTrainingTime:.3f}")
#print(f"Average ReLU testing runtime: {ReLUMeanTestingTime:.3f}")
print()

#Show Mish metrics for SGD on ResNet:
MishTestingAccuracy = stats.mean([CIFAR10_ResNet_SGD_Trials[i][47] for i in range(5)])
MishConverganceTestingStdv = stats.stdev([CIFAR10_ResNet_SGD_Trials[i][47] for i in range(5)])

MishConverganceValidationAccuracy = stats.mean([np.max(CIFAR10_ResNet_SGD_Trials[i][12]) for i in range(5)])
MishConverganceValidationMedian = stats.median([np.max(CIFAR10_ResNet_SGD_Trials[i][12]) for i in range(5)])
MishConverganceValidationStdv = stats.stdev([np.max(CIFAR10_ResNet_SGD_Trials[i][12]) for i in range(5)])

MishConverganceEpochIndeces = [np.argmax(CIFAR10_ResNet_SGD_Trials[i][12]) for i in range(5)]
MishConverganceEpochFloats = [float(MishConverganceEpochIndex) for MishConverganceEpochIndex in MishConverganceEpochIndeces]
MishConverganceEpochAverage = stats.mean(MishConverganceEpochFloats)

MishConverganceTrainingAccuracy = stats.mean([CIFAR10_ResNet_SGD_Trials[i][5][MishConverganceEpochIndeces[i]] for i in range(5)])

MishFittingIndex = (100-MishConverganceTrainingAccuracy)/(100-MishTestingAccuracy)

MishMinTrainingError = stats.mean([np.min(CIFAR10_ResNet_SGD_Trials[i][19]) for i in range(5)])

MishMeanTrainingTime = stats.mean([stats.mean(CIFAR10_ResNet_SGD_Trials[i][33]) for i in range(5)])

MishMeanValidationTime = stats.mean([stats.mean(CIFAR10_ResNet_SGD_Trials[i][40]) for i in range(5)])

print(f"Average Mish convergance Training accuracy: {MishConverganceTrainingAccuracy:.3f}")
print(f"Average Mish convergance Validation accuracy: {MishConverganceValidationAccuracy:.3f}")
print(f"Average Mish convergance Testing accuracy: {MishTestingAccuracy:.3f}")
#print(f"Median Mish convergance Testing accuracy: {MishConverganceTestingMedian:.3f}")
#print(f"Stdv of Mish convergance Validation accuracy: {MishConverganceValidationStdv:.3f}")
print(f"Average Mish convergance epoch: {MishConverganceEpochAverage:.3f}")
print(f"Average Mish fitting index: {MishFittingIndex:.3f}")
print(f"Stdv of Mish convergance test accuracy: {MishConverganceTestingStdv:.3f}")
#print(f"Average Mish minimum training error: {MishMinTrainingError:.3f}")
#print(f"Average Mish epoch runtime: {MishMeanTrainingTime:.3f}")
#print(f"Average Mish testing runtime: {MishMeanTestingTime:.3f}")
print()

#Show Swish metrics for SGD on ResNet:
SwishTestingAccuracy = stats.mean([CIFAR10_ResNet_SGD_Trials[i][48] for i in range(5)])
SwishConverganceTestingStdv = stats.stdev([CIFAR10_ResNet_SGD_Trials[i][48] for i in range(5)])

SwishConverganceValidationAccuracy = stats.mean([np.max(CIFAR10_ResNet_SGD_Trials[i][13]) for i in range(5)])
SwishConverganceValidationMedian = stats.median([np.max(CIFAR10_ResNet_SGD_Trials[i][13]) for i in range(5)])
SwishConverganceValidationStdv = stats.stdev([np.max(CIFAR10_ResNet_SGD_Trials[i][13]) for i in range(5)])

SwishConverganceEpochIndeces = [np.argmax(CIFAR10_ResNet_SGD_Trials[i][13]) for i in range(5)]
SwishConverganceEpochFloats = [float(SwishConverganceEpochIndex) for SwishConverganceEpochIndex in SwishConverganceEpochIndeces]
SwishConverganceEpochAverage = stats.mean(SwishConverganceEpochFloats)

SwishConverganceTrainingAccuracy = stats.mean([CIFAR10_ResNet_SGD_Trials[i][6][SwishConverganceEpochIndeces[i]] for i in range(5)])

SwishFittingIndex = (100-SwishConverganceTrainingAccuracy)/(100-SwishTestingAccuracy)

SwishMinTrainingError = stats.mean([np.min(CIFAR10_ResNet_SGD_Trials[i][20]) for i in range(5)])

SwishMeanTrainingTime = stats.mean([stats.mean(CIFAR10_ResNet_SGD_Trials[i][34]) for i in range(5)])

SwishMeanValidationTime = stats.mean([stats.mean(CIFAR10_ResNet_SGD_Trials[i][41]) for i in range(5)])

print(f"Average Swish convergance Training accuracy: {SwishConverganceTrainingAccuracy:.3f}")
print(f"Average Swish convergance Validation accuracy: {SwishConverganceValidationAccuracy:.3f}")
print(f"Average Swish convergance Testing accuracy: {SwishTestingAccuracy:.3f}")
#print(f"Median Swish convergance Testing accuracy: {SwishConverganceTestingMedian:.3f}")
#print(f"Stdv of Swish convergance Validation accuracy: {SwishConverganceValidationStdv:.3f}")
print(f"Average Swish convergance epoch: {SwishConverganceEpochAverage:.3f}")
print(f"Average Swish fitting index: {SwishFittingIndex:.3f}")
print(f"Stdv of Swish convergance test accuracy: {SwishConverganceTestingStdv:.3f}")
#print(f"Average Swish minimum training error: {SwishMinTrainingError:.3f}")
#print(f"Average Swish epoch runtime: {SwishMeanTrainingTime:.3f}")
#print(f"Average Swish testing runtime: {SwishMeanTestingTime:.3f}")
print()