#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 20:55:00 2024

@author: petersonco
"""

import argparse
import datetime
from logger import logging
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from dsets import LunaDataset
from torch.utils.data import DataLoader
from model import LunaModel
import numpy as np
from utils import enumerateWithEstimate

METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE = 3

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class LunaTrainingApp:
    

    
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys_argv[1:]
        print("hello")
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )

        parser.add_argument('--tb-prefix',
            default='p2ch11',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='dwlpt',
        )
        
        self.cli_args = parser.parse_args(sys_argv)
        print(self.cli_args)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        
        self.use_mps = torch.backends.mps.is_available()
        self.device = torch.device('cpu')
        self.totalTrainingSamples_count = 0
        
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
    
    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
        train_ds = self.initTrain()
        
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            trnMetrics_t = self.doTraining(epoch_ndx, train_ds)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)
    
    def initModel(self):
        model = LunaModel()
        model = model.to(self.device)
        return model
    
    def initOptimizer(self):
        return SGD(self.model.parameters(), lr = 0.001, momentum=0.99)
    
    def initTrain(self):
        train_ds = LunaDataset(val_stride=10,isValSet_bool=False)
        batch_size = self.cli_args.batch_size
        loader = DataLoader(train_ds,batch_size=batch_size,num_workers=self.cli_args.num_workers,
                            pin_memory=self.use_mps)
        return loader
    
    def initValDl(self):
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
        )

        batch_size = self.cli_args.batch_size

        loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_mps,
        )
        return loader
   
    def doTraining(self, epoch_ndx, train_ds):
        self.model.train()
        
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_ds.dataset),
            device=self.device,
        )
        
        batch_iter = enumerateWithEstimate(
            train_ds,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_ds.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_ds.batch_size,
                trnMetrics_g
            )
        
            loss_var.backward()
            self.optimizer.step()
        
        self.totalTrainingSamples_count += len(train_ds.dataset)

        return trnMetrics_g.to('cpu')
    
    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')
        
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup
        
        inputs = input_t.to(self.device, non_blocking=True)
        labels = label_t.to(self.device, non_blocking=True)
        
        output_values, probability_values = self.model(inputs)
        
        loss_func = nn.CrossEntropyLoss(reduction='none')
    
        loss_val = loss_func(output_values,labels[:,1],)
        
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
        labels[:,1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
        probability_values[:,1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
        loss_val.detach() 

        return loss_val.mean()
    
    def logMetrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
            classificationThreshold=0.5,
    ):
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold
        
        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask
        
        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())
        
        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())
        
        metrics_dict = {}
        metrics_dict['loss/all'] = \
            metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = \
            metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = \
            metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()
        
        metrics_dict['correct/all'] = (pos_correct + neg_correct) \
            / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100
        
        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
                 + "{correct/all:-5.1f}% correct, "
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
                 + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
                 + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )
        
        writer = getattr(self, mode_str + '_writer')
        
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)
        
        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )
        
        bins = [x/50.0 for x in range(51)]
        
        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)
        
        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        
if __name__ == '__main__':
    LunaTrainingApp().main()