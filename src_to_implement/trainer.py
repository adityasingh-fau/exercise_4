import numpy as np
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self.targetCracks = []
        self.targetInactive = []
        self.outCracks = []
        self.outInactive = []

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps: -reset the gradients. By default, PyTorch accumulates (sums up) gradients when
        # backward() is called. This behavior is not required here, so you need to ensure that all the gradients are
        # zero before calling the backward.
        self._optim.zero_grad()
        # -propagate through the network
        prop = self._model(x)
        # -calculate the loss
        lossOutput = self._crit(prop, y)
        # -compute gradient by backward propagation
        lossOutput.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return lossOutput

    def val_test_step(self, x, y):
        # predict
        output = self._model(x)
        # propagate through the network and calculate the loss and predictions
        loss = self._crit(output, y)
        # return the loss and the predictions
        return loss, output

    def train_epoch(self):
        # set training mode
        self._model.train()
        # iterate through the training set.
        loss = 0.0
        total_data = 0
        for i, data in tqdm(enumerate(self._train_dl, 0)):
            inputs, labels = data
            total_data = len(data)
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            # perform a training step
            loss += self.train_step(inputs, labels)
            # calculate the average loss for the epoch and return it
        return loss / total_data

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout,
        # BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        losses = Metric('Loss',':.4e')
        self._model.eval()
        predictions = []
        storedLabels = []
        batchLen = len(self._val_test_dl)
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't
        # required anymore.
        with t.no_grad():
            # iterate through the validation set
            running_loss = 0.0
            total_data = 0
            for i, data in tqdm(enumerate(self._val_test_dl, 0)):
                inputs, labels = data
                total_data += len(data)
                # transfer the batch to the gpu if given
                if self._cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                # perform a validation step
                loss, prediction = self.val_test_step(inputs, labels)
                losses.update(loss, batchLen)
                # save the predictions and the labels for each batch
                predictions.append(prediction)
                storedLabels.append(labels)

        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        avg_loss = running_loss / batchLen
        # print(storedLabels)
        # f1score = f1_score(storedLabels, predictions, average=None)
        # print(f1score)

        # return the loss and print the calculated metrics
        self.crackF1score = []
        self.inactiveF1score = []

        for i in range(batchLen):
            predictValue = ((predictions[i]>0.5).cpu() * t.tensor([1]))
            labelValue = ((storedLabels[i]).cpu() * t.tensor([1]))
            predictCrackValue = predictValue[:,0]
            predictInactiveValue = predictValue[:,1]
            labelCrackValue = labelValue[:,0]
            labelInactiveValue = labelValue[:,1]
            self.targetCracks += labelCrackValue.tolist()
            self.targetInactive += labelInactiveValue.tolist()
            self.outCracks += predictCrackValue.tolist()
            self.outInactive += predictInactiveValue.tolist()

            avgMethod = 'macro'  #Manipulate
            crackF1Score = f1_score(labelCrackValue,predictCrackValue, average= avgMethod)
            inactiveF1Score = f1_score(labelInactiveValue,predictInactiveValue,average = avgMethod)
            self.crackF1score.append(crackF1Score)
            self.inactiveF1score.append(inactiveF1Score)
        avg_loss = losses.average()
        # f1_mean, f1_cracks_mean, f1_inactives_mean = self.f1_scores()
        # print("Test: F1 Crack {} F1 Inactive {} F1 Mean {}".format(f1_cracks_mean, f1_inactives_mean, f1_mean))
        return avg_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        train_Loss = []
        val_Loss = []
        epoch = 0

        while True:

            # stop by epoch number
            if epoch > epochs:
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            trainloss = self.train_epoch()
            valloss = self.val_test()

            self.crackF1scoreFit = []
            self.inactiveF1scoreFit = []
            avgMethod = 'macro'  # Manipulate
            crackF1Score = f1_score(self.targetCracks, self.outCracks, average=avgMethod)
            inactiveF1Score = f1_score(self.targetInactive, self.outInactive, average=avgMethod)
            self.crackF1scoreFit.append(crackF1Score)
            self.inactiveF1scoreFit.append(inactiveF1Score)
            # f1_mean_fit, f1_crack_fit, f1_inactive_fit = self.f1_scoresFit()
            # print("Fit:  F1 Crack {} F1 Inactive {} F1 Mean {}".format(f1_mean_fit, f1_inactive_fit, f1_mean_fit))
            # append the losses to the respective lists
            train_Loss.append(trainloss)
            val_Loss.append(valloss)
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            self.save_checkpoint(epoch)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if epoch > 0:
                self._early_stopping_patience = val_Loss[epoch - 1] - val_Loss[epoch]
            # return the losses for both training and validation
            epoch += 1
        return train_Loss, val_Loss

    def f1_scores(self):
        f1scoreCrack = np.array(self.crackF1score)
        f1scoreInactive = np.array(self.inactiveF1score)
        meanF1crack = np.mean(f1scoreCrack)
        meanF1inactive = np.mean(f1scoreInactive)
        meanF1 = np.mean((meanF1crack, meanF1inactive))
        return meanF1, meanF1crack, meanF1inactive

    def f1_scoresFit(self):
        f1scoreCrack = np.array(self.crackF1scoreFit)
        f1scoreInactive = np.array(self.inactiveF1scoreFit)
        meanF1crack = np.mean(f1scoreCrack)
        meanF1inactive = np.mean(f1scoreInactive)
        meanF1 = np.mean((meanF1crack, meanF1inactive))
        return meanF1, meanF1crack, meanF1inactive

    class Metrics:

        def __init__(self, name, fmt=':f'):
            self.name = name
            self.fmt = fmt
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def __str__(self):
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

        def average(self):
            return self.avg
