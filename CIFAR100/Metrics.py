import statistics as stats
import numpy as np
from math import isnan



CIFAR100_ResNet_SGD_Trials = []
CIFAR100_ResNet_SGD_Trials = []
CIFAR100_ResNet_SGD_Trials = []
CIFAR100_ResNet_SGD_Trials = []

CIFAR100_ResNet_SGD_Trials = []
CIFAR100_ResNet_SGD_Trials = []
CIFAR100_ResNet_SGD_Trials = []
CIFAR100_ResNet_SGD_Trials = []

CIFAR100_WideResNet_SGD_Trials = []
CIFAR100_WideResNet_SGD_Trials = []
CIFAR100_WideResNet_SGD_Trials = []
CIFAR100_WideResNet_SGD_Trials = []

#Read ResNet SGD Trials
for i in range(1,6):
  tempFile = open('ResNet50/SGD/ResNet_CIFAR100_Trial'+str(i)+'.csv', 'r')
  tempTrial = []

  #Read lines
  SishTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  SishValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  SishTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  SishValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  SishTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  SishValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  SishTestingAccuracy = float(tempFile.readline().replace("\"",""))
  GELUTestingAccuracy = float(tempFile.readline().replace("\"",""))
  ReLUTestingAccuracy = float(tempFile.readline().replace("\"",""))

  #append CSV lines to temp trial list of metrics
  tempTrial.append(SishTrainingAccuracy)
  tempTrial.append(GELUTrainingAccuracy)
  tempTrial.append(ReLUTrainingAccuracy)

  tempTrial.append(SishValidationAccuracy)
  tempTrial.append(GELUValidationAccuracy)
  tempTrial.append(ReLUValidationAccuracy)

  tempTrial.append(SishTrainingError)
  tempTrial.append(GELUTrainingError)
  tempTrial.append(ReLUTrainingError)

  tempTrial.append(SishValidationError)
  tempTrial.append(GELUValidationError)
  tempTrial.append(ReLUValidationError)

  tempTrial.append(SishTrainingRuntime)
  tempTrial.append(GELUTrainingRuntime)
  tempTrial.append(ReLUTrainingRuntime)

  tempTrial.append(SishValidationRuntime)
  tempTrial.append(GELUValidationRuntime)
  tempTrial.append(ReLUValidationRuntime)

  tempTrial.append(SishTestingAccuracy)
  tempTrial.append(GELUTestingAccuracy)
  tempTrial.append(ReLUTestingAccuracy)

  #append trial metric lists to trials lists
  CIFAR100_ResNet_SGD_Trials.append(tempTrial)
  tempFile.close()

#Prepare data from not-a-number values #REWORK??? todo: todo
for i in range(len(CIFAR100_ResNet_SGD_Trials)):
  for j in range(len(CIFAR100_ResNet_SGD_Trials[i][6])):
    if isnan(CIFAR100_ResNet_SGD_Trials[i][6][j]):
      CIFAR100_ResNet_SGD_Trials[i][14][j] = 1000.
    if isnan(CIFAR100_ResNet_SGD_Trials[i][7][j]):
      CIFAR100_ResNet_SGD_Trials[i][15][j] = 1000.
    if isnan(CIFAR100_ResNet_SGD_Trials[i][8][j]):
      CIFAR100_ResNet_SGD_Trials[i][16][j] = 1000.

print("ResNet SGD:")
print()

#Show Sish metrics for SGD on ResNet:
# for i in range(5):
#   print(CIFAR100_ResNet_SGD_Trials[i][43])
SishTestingAccuracy = stats.mean([CIFAR100_ResNet_SGD_Trials[i][18] for i in range(5)])
SishConverganceTestingStdv = stats.stdev([CIFAR100_ResNet_SGD_Trials[i][18] for i in range(5)])

SishConverganceValidationAccuracy = stats.mean([np.max(CIFAR100_ResNet_SGD_Trials[i][3]) for i in range(5)])
SishConverganceValidationMedian = stats.median([np.max(CIFAR100_ResNet_SGD_Trials[i][3]) for i in range(5)])
SishConverganceValidationStdv = stats.stdev([np.max(CIFAR100_ResNet_SGD_Trials[i][3]) for i in range(5)])

SishConclusiveValidationAccuracy = stats.mean([CIFAR100_ResNet_SGD_Trials[i][3][-1] for i in range(5)])
SishStabilityIndex = SishConclusiveValidationAccuracy / SishConverganceValidationAccuracy

SishConverganceEpochIndeces = [np.argmax(CIFAR100_ResNet_SGD_Trials[i][3]) for i in range(5)]
SishConverganceEpochFloats = [float(SishConverganceEpochIndex) for SishConverganceEpochIndex in SishConverganceEpochIndeces]
SishConverganceEpochAverage = stats.mean(SishConverganceEpochFloats)

SishConverganceTrainingAccuracy = stats.mean([CIFAR100_ResNet_SGD_Trials[i][0][SishConverganceEpochIndeces[i]] for i in range(5)])

SishFittingIndex = (100-SishConverganceTrainingAccuracy)/(100-SishTestingAccuracy)

SishMinTrainingError = stats.mean([np.min(CIFAR100_ResNet_SGD_Trials[i][6]) for i in range(5)])

SishMeanTrainingTime = stats.mean([stats.mean(CIFAR100_ResNet_SGD_Trials[i][12]) for i in range(5)])

SishMeanValidationTime = stats.mean([stats.mean(CIFAR100_ResNet_SGD_Trials[i][15]) for i in range(5)])

print(f"Average Sish convergance Training accuracy: {SishConverganceTrainingAccuracy:.3f}")
print(f"Average Sish convergance Validation accuracy: {SishConverganceValidationAccuracy:.3f}")
print(f"Average Sish convergance Testing accuracy: {SishTestingAccuracy:.3f}")
#print(f"Median Sish convergance Testing accuracy: {SishConverganceTestingMedian:.3f}")
#print(f"Stdv of Sish convergance Validation accuracy: {SishConverganceValidationStdv:.3f}")
# print(f"Average Sish convergance epoch: {SishConverganceEpochAverage:.3f}")
#print(f"Average Sish fitting index: {SishFittingIndex:.3f}")
print(f"Average Sish conclusive Validation accuracy: {SishConclusiveValidationAccuracy:.3f}")
print(f"Stdv of Sish convergance test accuracy: {SishConverganceTestingStdv:.3f}")
#print(f"Average Sish minimum training error: {SishMinTrainingError:.3f}")
#print(f"Average Sish epoch runtime: {SishMeanTrainingTime:.3f}")
#print(f"Average Sish testing runtime: {SishMeanTestingTime:.3f}")
print()

#Show GELU metrics for SGD on ResNet:
GELUTestingAccuracy = stats.mean([CIFAR100_ResNet_SGD_Trials[i][19] for i in range(5)])
GELUConverganceTestingStdv = stats.stdev([CIFAR100_ResNet_SGD_Trials[i][19] for i in range(5)])

GELUConverganceValidationAccuracy = stats.mean([np.max(CIFAR100_ResNet_SGD_Trials[i][4]) for i in range(5)])
GELUConverganceValidationMedian = stats.median([np.max(CIFAR100_ResNet_SGD_Trials[i][4]) for i in range(5)])
GELUConverganceValidationStdv = stats.stdev([np.max(CIFAR100_ResNet_SGD_Trials[i][4]) for i in range(5)])

GELUConclusiveValidationAccuracy = stats.mean([CIFAR100_ResNet_SGD_Trials[i][4][-1] for i in range(5)])
GELUStabilityIndex = GELUConclusiveValidationAccuracy / GELUConverganceValidationAccuracy

GELUConverganceEpochIndeces = [np.argmax(CIFAR100_ResNet_SGD_Trials[i][4]) for i in range(5)]
GELUConverganceEpochFloats = [float(GELUConverganceEpochIndex) for GELUConverganceEpochIndex in GELUConverganceEpochIndeces]
GELUConverganceEpochAverage = stats.mean(GELUConverganceEpochFloats)

GELUConverganceTrainingAccuracy = stats.mean([CIFAR100_ResNet_SGD_Trials[i][1][GELUConverganceEpochIndeces[i]] for i in range(5)])

GELUFittingIndex = (100-GELUConverganceTrainingAccuracy)/(100-GELUTestingAccuracy)

GELUMinTrainingError = stats.mean([np.min(CIFAR100_ResNet_SGD_Trials[i][7]) for i in range(5)])

GELUMeanTrainingTime = stats.mean([stats.mean(CIFAR100_ResNet_SGD_Trials[i][13]) for i in range(5)])

GELUMeanValidationTime = stats.mean([stats.mean(CIFAR100_ResNet_SGD_Trials[i][16]) for i in range(5)])

print(f"Average GELU convergance Training accuracy: {GELUConverganceTrainingAccuracy:.3f}")
print(f"Average GELU convergance Validation accuracy: {GELUConverganceValidationAccuracy:.3f}")
print(f"Average GELU convergance Testing accuracy: {GELUTestingAccuracy:.3f}")
#print(f"Median GELU convergance Testing accuracy: {GELUConverganceTestingMedian:.3f}")
#print(f"Stdv of GELU convergance Validation accuracy: {GELUConverganceValidationStdv:.3f}")
# print(f"Average GELU convergance epoch: {GELUConverganceEpochAverage:.3f}")
#print(f"Average GELU fitting index: {GELUFittingIndex:.3f}")
print(f"Average GELU conclusive Validation accuracy: {GELUConclusiveValidationAccuracy:.3f}")
print(f"Stdv of GELU convergance test accuracy: {GELUConverganceTestingStdv:.3f}")
#print(f"Average GELU minimum training error: {GELUMinTrainingError:.3f}")
#print(f"Average GELU epoch runtime: {GELUMeanTrainingTime:.3f}")
#print(f"Average GELU testing runtime: {GELUMeanTestingTime:.3f}")
print()


#Show ReLU metrics for SGD on ResNet:
ReLUTestingAccuracy = stats.mean([CIFAR100_ResNet_SGD_Trials[i][20] for i in range(5)])
ReLUConverganceTestingStdv = stats.stdev([CIFAR100_ResNet_SGD_Trials[i][20] for i in range(5)])

ReLUConverganceValidationAccuracy = stats.mean([np.max(CIFAR100_ResNet_SGD_Trials[i][5]) for i in range(5)])
ReLUConverganceValidationMedian = stats.median([np.max(CIFAR100_ResNet_SGD_Trials[i][5]) for i in range(5)])
ReLUConverganceValidationStdv = stats.stdev([np.max(CIFAR100_ResNet_SGD_Trials[i][5]) for i in range(5)])

ReLUConclusiveValidationAccuracy = stats.mean([CIFAR100_ResNet_SGD_Trials[i][5][-1] for i in range(5)])
ReLUStabilityIndex = ReLUConclusiveValidationAccuracy / ReLUConverganceValidationAccuracy

ReLUConverganceEpochIndeces = [np.argmax(CIFAR100_ResNet_SGD_Trials[i][5]) for i in range(5)]
ReLUConverganceEpochFloats = [float(ReLUConverganceEpochIndex) for ReLUConverganceEpochIndex in ReLUConverganceEpochIndeces]
ReLUConverganceEpochAverage = stats.mean(ReLUConverganceEpochFloats)

ReLUConverganceTrainingAccuracy = stats.mean([CIFAR100_ResNet_SGD_Trials[i][2][ReLUConverganceEpochIndeces[i]] for i in range(5)])

ReLUFittingIndex = (100-ReLUConverganceTrainingAccuracy)/(100-ReLUTestingAccuracy)

ReLUMinTrainingError = stats.mean([np.min(CIFAR100_ResNet_SGD_Trials[i][8]) for i in range(5)])

ReLUMeanTrainingTime = stats.mean([stats.mean(CIFAR100_ResNet_SGD_Trials[i][14]) for i in range(5)])

ReLUMeanValidationTime = stats.mean([stats.mean(CIFAR100_ResNet_SGD_Trials[i][17]) for i in range(5)])

print(f"Average ReLU convergance Training accuracy: {ReLUConverganceTrainingAccuracy:.3f}")
print(f"Average ReLU convergance Validation accuracy: {ReLUConverganceValidationAccuracy:.3f}")
print(f"Average ReLU convergance Testing accuracy: {ReLUTestingAccuracy:.3f}")
#print(f"Median ReLU convergance Testing accuracy: {ReLUConverganceTestingMedian:.3f}")
#print(f"Stdv of ReLU convergance Validation accuracy: {ReLUConverganceValidationStdv:.3f}")
# print(f"Average ReLU convergance epoch: {ReLUConverganceEpochAverage:.3f}")
#print(f"Average ReLU fitting index: {ReLUFittingIndex:.3f}")
print(f"Average ReLU conclusive Validation accuracy: {ReLUConclusiveValidationAccuracy:.3f}")
print(f"Stdv of ReLU convergance test accuracy: {ReLUConverganceTestingStdv:.3f}")
#print(f"Average ReLU minimum training error: {ReLUMinTrainingError:.3f}")
#print(f"Average ReLU epoch runtime: {ReLUMeanTrainingTime:.3f}")
#print(f"Average ReLU testing runtime: {ReLUMeanTestingTime:.3f}")
print()