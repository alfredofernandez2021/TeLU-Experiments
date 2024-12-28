import statistics as stats
import numpy as np
from math import isnan

CIFAR10_DenseNet_SGD_Trials = []
CIFAR10_DenseNet_SGD_Trials = []
CIFAR10_DenseNet_SGD_Trials = []
CIFAR10_DenseNet_SGD_Trials = []

CIFAR10_DenseNet_SGD_Trials = []
CIFAR10_DenseNet_SGD_Trials = []
CIFAR10_DenseNet_SGD_Trials = []
CIFAR10_DenseNet_SGD_Trials = []

CIFAR10_WideDenseNet_SGD_Trials = []
CIFAR10_WideDenseNet_SGD_Trials = []
CIFAR10_WideDenseNet_SGD_Trials = []
CIFAR10_WideDenseNet_SGD_Trials = []

#Read DenseNet SGD Trials
for i in range(0,10):
  tempFile = open('DenseNet_CIFAR10_Trial'+str(i)+'.csv', 'r')
  tempTrial = []

  #Read lines
  TeLUTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  TeLUValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  TeLUTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  TeLUValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  TeLUTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  TeLUValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  GELUValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
  ReLUValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  TeLUTestingAccuracy = float(tempFile.readline().replace("\"",""))
  GELUTestingAccuracy = float(tempFile.readline().replace("\"",""))
  ReLUTestingAccuracy = float(tempFile.readline().replace("\"",""))

  #append CSV lines to temp trial list of metrics
  tempTrial.append(TeLUTrainingAccuracy)
  tempTrial.append(GELUTrainingAccuracy)
  tempTrial.append(ReLUTrainingAccuracy)

  tempTrial.append(TeLUValidationAccuracy)
  tempTrial.append(GELUValidationAccuracy)
  tempTrial.append(ReLUValidationAccuracy)

  tempTrial.append(TeLUTrainingError)
  tempTrial.append(GELUTrainingError)
  tempTrial.append(ReLUTrainingError)

  tempTrial.append(TeLUValidationError)
  tempTrial.append(GELUValidationError)
  tempTrial.append(ReLUValidationError)

  tempTrial.append(TeLUTrainingRuntime)
  tempTrial.append(GELUTrainingRuntime)
  tempTrial.append(ReLUTrainingRuntime)

  tempTrial.append(TeLUValidationRuntime)
  tempTrial.append(GELUValidationRuntime)
  tempTrial.append(ReLUValidationRuntime)

  tempTrial.append(TeLUTestingAccuracy)
  tempTrial.append(GELUTestingAccuracy)
  tempTrial.append(ReLUTestingAccuracy)

  #append trial metric lists to trials lists
  CIFAR10_DenseNet_SGD_Trials.append(tempTrial)
  tempFile.close()

#Prepare data from not-a-number values #REWORK??? todo: todo
# for i in range(len(CIFAR10_DenseNet_SGD_Trials)):
#   for j in range(len(CIFAR10_DenseNet_SGD_Trials[i][14])):
#     if isnan(CIFAR10_DenseNet_SGD_Trials[i][14][j]):
#       CIFAR10_DenseNet_SGD_Trials[i][14][j] = 1000.
#     if isnan(CIFAR10_DenseNet_SGD_Trials[i][15][j]):
#       CIFAR10_DenseNet_SGD_Trials[i][15][j] = 1000.
#     if isnan(CIFAR10_DenseNet_SGD_Trials[i][16][j]):
#       CIFAR10_DenseNet_SGD_Trials[i][16][j] = 1000.
#     if isnan(CIFAR10_DenseNet_SGD_Trials[i][17][j]):
#       CIFAR10_DenseNet_SGD_Trials[i][17][j] = 1000.
#     if isnan(CIFAR10_DenseNet_SGD_Trials[i][18][j]):
#       CIFAR10_DenseNet_SGD_Trials[i][18][j] = 1000.
#     if isnan(CIFAR10_DenseNet_SGD_Trials[i][19][j]):
#       CIFAR10_DenseNet_SGD_Trials[i][19][j] = 1000.
#     if isnan(CIFAR10_DenseNet_SGD_Trials[i][20][j]):
#       CIFAR10_DenseNet_SGD_Trials[i][20][j] = 1000.

print("DenseNet SGD:")
print()

#Show TeLU metrics for SGD on DenseNet:
# for i in range(10):
#   print(CIFAR10_DenseNet_SGD_Trials[i][43])
TeLUTestingAccuracy = stats.mean([CIFAR10_DenseNet_SGD_Trials[i][18] for i in range(10)])
TeLUConverganceTestingStdv = stats.stdev([CIFAR10_DenseNet_SGD_Trials[i][18] for i in range(10)])

TeLUConverganceValidationAccuracy = stats.mean([np.max(CIFAR10_DenseNet_SGD_Trials[i][3]) for i in range(10)])
TeLUConverganceValidationMedian = stats.median([np.max(CIFAR10_DenseNet_SGD_Trials[i][3]) for i in range(10)])
TeLUConverganceValidationStdv = stats.stdev([np.max(CIFAR10_DenseNet_SGD_Trials[i][3]) for i in range(10)])

TeLUConverganceEpochIndeces = [np.argmax(CIFAR10_DenseNet_SGD_Trials[i][3]) for i in range(10)]
TeLUConverganceEpochFloats = [float(TeLUConverganceEpochIndex) for TeLUConverganceEpochIndex in TeLUConverganceEpochIndeces]
TeLUConverganceEpochAverage = stats.mean(TeLUConverganceEpochFloats)

TeLUConverganceTrainingAccuracy = stats.mean([CIFAR10_DenseNet_SGD_Trials[i][0][TeLUConverganceEpochIndeces[i]] for i in range(10)])

TeLUFittingIndex = (100-TeLUConverganceTrainingAccuracy)/(100-TeLUTestingAccuracy)

TeLUMinTrainingError = stats.mean([np.min(CIFAR10_DenseNet_SGD_Trials[i][6]) for i in range(10)])

TeLUMeanTrainingTime = stats.mean([stats.mean(CIFAR10_DenseNet_SGD_Trials[i][12]) for i in range(10)])

TeLUMeanValidationTime = stats.mean([stats.mean(CIFAR10_DenseNet_SGD_Trials[i][15]) for i in range(10)])

print(f"Average TeLU convergance Training accuracy: {TeLUConverganceTrainingAccuracy:.3f}")
print(f"Average TeLU convergance Validation accuracy: {TeLUConverganceValidationAccuracy:.3f}")
print(f"Average TeLU convergance Testing accuracy: {TeLUTestingAccuracy:.3f}")
#print(f"Median TeLU convergance Testing accuracy: {TeLUConverganceTestingMedian:.3f}")
print(f"Stdv of TeLU convergance Validation accuracy: {TeLUConverganceValidationStdv:.3f}")
print(f"Average TeLU convergance epoch: {TeLUConverganceEpochAverage:.3f}")
print(f"Average TeLU fitting index: {TeLUFittingIndex:.3f}")
print(f"Stdv of TeLU convergance test accuracy: {TeLUConverganceTestingStdv:.3f}")
#print(f"Average TeLU minimum training error: {TeLUMinTrainingError:.3f}")
#print(f"Average TeLU epoch runtime: {TeLUMeanTrainingTime:.3f}")
#print(f"Average TeLU testing runtime: {TeLUMeanTestingTime:.3f}")
print()

#Show GELU metrics for SGD on DenseNet:
GELUTestingAccuracy = stats.mean([CIFAR10_DenseNet_SGD_Trials[i][19] for i in range(10)])
GELUConverganceTestingStdv = stats.stdev([CIFAR10_DenseNet_SGD_Trials[i][19] for i in range(10)])

GELUConverganceValidationAccuracy = stats.mean([np.max(CIFAR10_DenseNet_SGD_Trials[i][4]) for i in range(10)])
GELUConverganceValidationMedian = stats.median([np.max(CIFAR10_DenseNet_SGD_Trials[i][4]) for i in range(10)])
GELUConverganceValidationStdv = stats.stdev([np.max(CIFAR10_DenseNet_SGD_Trials[i][4]) for i in range(10)])

GELUConverganceEpochIndeces = [np.argmax(CIFAR10_DenseNet_SGD_Trials[i][4]) for i in range(10)]
GELUConverganceEpochFloats = [float(GELUConverganceEpochIndex) for GELUConverganceEpochIndex in GELUConverganceEpochIndeces]
GELUConverganceEpochAverage = stats.mean(GELUConverganceEpochFloats)

GELUConverganceTrainingAccuracy = stats.mean([CIFAR10_DenseNet_SGD_Trials[i][1][GELUConverganceEpochIndeces[i]] for i in range(10)])

GELUFittingIndex = (100-GELUConverganceTrainingAccuracy)/(100-GELUTestingAccuracy)

GELUMinTrainingError = stats.mean([np.min(CIFAR10_DenseNet_SGD_Trials[i][7]) for i in range(10)])

GELUMeanTrainingTime = stats.mean([stats.mean(CIFAR10_DenseNet_SGD_Trials[i][13]) for i in range(10)])

GELUMeanValidationTime = stats.mean([stats.mean(CIFAR10_DenseNet_SGD_Trials[i][16]) for i in range(10)])

print(f"Average GELU convergance Training accuracy: {GELUConverganceTrainingAccuracy:.3f}")
print(f"Average GELU convergance Validation accuracy: {GELUConverganceValidationAccuracy:.3f}")
print(f"Average GELU convergance Testing accuracy: {GELUTestingAccuracy:.3f}")
#print(f"Median GELU convergance Testing accuracy: {GELUConverganceTestingMedian:.3f}")
print(f"Stdv of GELU convergance Validation accuracy: {GELUConverganceValidationStdv:.3f}")
print(f"Average GELU convergance epoch: {GELUConverganceEpochAverage:.3f}")
print(f"Average GELU fitting index: {GELUFittingIndex:.3f}")
print(f"Stdv of GELU convergance test accuracy: {GELUConverganceTestingStdv:.3f}")
#print(f"Average GELU minimum training error: {GELUMinTrainingError:.3f}")
#print(f"Average GELU epoch runtime: {GELUMeanTrainingTime:.3f}")
#print(f"Average GELU testing runtime: {GELUMeanTestingTime:.3f}")
print()


#Show ReLU metrics for SGD on DenseNet:
ReLUTestingAccuracy = stats.mean([CIFAR10_DenseNet_SGD_Trials[i][20] for i in range(10)])
ReLUConverganceTestingStdv = stats.stdev([CIFAR10_DenseNet_SGD_Trials[i][20] for i in range(10)])

ReLUConverganceValidationAccuracy = stats.mean([np.max(CIFAR10_DenseNet_SGD_Trials[i][5]) for i in range(10)])
ReLUConverganceValidationMedian = stats.median([np.max(CIFAR10_DenseNet_SGD_Trials[i][5]) for i in range(10)])
ReLUConverganceValidationStdv = stats.stdev([np.max(CIFAR10_DenseNet_SGD_Trials[i][5]) for i in range(10)])

ReLUConverganceEpochIndeces = [np.argmax(CIFAR10_DenseNet_SGD_Trials[i][5]) for i in range(10)]
ReLUConverganceEpochFloats = [float(ReLUConverganceEpochIndex) for ReLUConverganceEpochIndex in ReLUConverganceEpochIndeces]
ReLUConverganceEpochAverage = stats.mean(ReLUConverganceEpochFloats)

ReLUConverganceTrainingAccuracy = stats.mean([CIFAR10_DenseNet_SGD_Trials[i][2][ReLUConverganceEpochIndeces[i]] for i in range(10)])

ReLUFittingIndex = (100-ReLUConverganceTrainingAccuracy)/(100-ReLUTestingAccuracy)

ReLUMinTrainingError = stats.mean([np.min(CIFAR10_DenseNet_SGD_Trials[i][8]) for i in range(10)])

ReLUMeanTrainingTime = stats.mean([stats.mean(CIFAR10_DenseNet_SGD_Trials[i][14]) for i in range(10)])

ReLUMeanValidationTime = stats.mean([stats.mean(CIFAR10_DenseNet_SGD_Trials[i][17]) for i in range(10)])

print(f"Average ReLU convergance Training accuracy: {ReLUConverganceTrainingAccuracy:.3f}")
print(f"Average ReLU convergance Validation accuracy: {ReLUConverganceValidationAccuracy:.3f}")
print(f"Average ReLU convergance Testing accuracy: {ReLUTestingAccuracy:.3f}")
#print(f"Median ReLU convergance Testing accuracy: {ReLUConverganceTestingMedian:.3f}")
print(f"Stdv of ReLU convergance Validation accuracy: {ReLUConverganceValidationStdv:.3f}")
print(f"Average ReLU convergance epoch: {ReLUConverganceEpochAverage:.3f}")
print(f"Average ReLU fitting index: {ReLUFittingIndex:.3f}")
print(f"Stdv of ReLU convergance test accuracy: {ReLUConverganceTestingStdv:.3f}")
#print(f"Average ReLU minimum training error: {ReLUMinTrainingError:.3f}")
#print(f"Average ReLU epoch runtime: {ReLUMeanTrainingTime:.3f}")
#print(f"Average ReLU testing runtime: {ReLUMeanTestingTime:.3f}")
print()
