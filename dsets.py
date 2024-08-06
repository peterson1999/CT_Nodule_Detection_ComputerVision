#%%
from functools import lru_cache 
from collections import namedtuple
import glob
import os
import csv
import SimpleITK as sitk
import numpy as np
import collections
import utils
import torch
import copy
from torch.utils.data import Dataset


#%%
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)
#%%
class CT:
    def __init__(self,series_uid):
            mhd_path = glob.glob('/Users/petersonco/PycharmProjects/pythonProject1/Deep Learning with Pytorch/Part 2/LUNA Data/subset*/{}.mhd'.format(series_uid))[0]
            ct_mhd = sitk.ReadImage(mhd_path)
            ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype = np.float32)
            ct_a.clip(-1000,1000,ct_a)
            self.series_uid = series_uid
            self.hu_a = ct_a
            
            self.origin_xyz = utils.XYZTuple(*ct_mhd.GetOrigin())
            self.voxel_size_xyz = utils.XYZTuple(*ct_mhd.GetSpacing())
            self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3,3)
            
    def getCenterNodule(self,xyz_center,width):
        irc_center = utils.XYZtoIRC(xyz_center,self.origin_xyz, self.voxel_size_xyz,self.direction_a)
        slice_list = []
        for axis, coordinate in enumerate(irc_center):
            start_ndx = int(round(coordinate - (width[axis]/2)))
            end_ndx = int(start_ndx + width[axis])

            assert coordinate >= 0 and coordinate < self.hu_a.shape[axis], repr([self.series_uid, xyz_center, self.origin_xyz, self.voxel_size_xyz, irc_center, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width[axis])
            slice_list.append(slice(start_ndx,end_ndx))
        center_nodule_chunk = self.hu_a[tuple(slice_list)]
        return center_nodule_chunk, irc_center
#%%
@lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    mhd_list = glob.glob('/Users/petersonco/PycharmProjects/pythonProject1/Deep Learning with Pytorch/Part 2/LUNA Data/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict= {}
    
    with open('/Users/petersonco/PycharmProjects/pythonProject1/Deep Learning with Pytorch/Part 2/LUNA Data/annotations.csv',"r") as f:
        for row in (list(csv.reader(f))[1:]):
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            
            diameter_dict.setdefault(series_uid,[]).append((annotationCenter_xyz,annotationDiameter_mm))
    
    candidateInfo_list = []
    
    with open('/Users/petersonco/PycharmProjects/pythonProject1/Deep Learning with Pytorch/Part 2//LUNA Data/candidates.csv',"r") as f:
        for row in (list(csv.reader(f))[1:]):
            series_uid=row[0]
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                    continue
            
            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
            
            candidateDiameter_mm = 0.0
            
            for annotation_tup in diameter_dict.get(series_uid,[]):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                
                for i in range(3):
                    delta_mm = abs(annotationCenter_xyz[i] - candidateCenter_xyz[i])
                
                if delta_mm > annotationDiameter_mm/4:
                    break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break
                
            candidateInfo_list.append(CandidateInfoTuple(
              isNodule_bool, candidateDiameter_mm, series_uid, candidateCenter_xyz 
            ))
        candidateInfo_list.sort(reverse=True)
        print(len(candidateInfo_list))
    return candidateInfo_list

#%%
def getCt(series_uid):
    return  CT(series_uid)
#%%
def getCtCenterNodule(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    center_chunk, center_irc = ct.getCenterNodule(center_xyz,width_irc)
    return center_chunk, center_irc


#%%
class LunaDataset(Dataset):
    
    def __init__(self, val_stride=0, isValSet_bool=None,series_uid=None):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())
        assert self.candidateInfo_list
        if series_uid:
            self.candidateInfo_list = [x for x in self.candidateInfo_list if x.series_uid == series_uid]
            
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
            
    def __len__(self):
        return len(self.candidateInfo_list)
    
    def __getitem__(self,ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = [32,48,48]
        candidate_a, center_irc = getCtCenterNodule(candidateInfo_tup.series_uid,candidateInfo_tup.center_xyz,width_irc)
        
        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)
    
        pos_t = torch.tensor([not candidateInfo_tup.isNodule_bool, candidateInfo_tup.isNodule_bool],dtype=torch.long)
        
        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc),
        )
        

#55
