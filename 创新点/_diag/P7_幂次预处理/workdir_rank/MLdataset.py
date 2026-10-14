from torch.utils.data import Dataset, DataLoader
import scipy.io
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, scale
import math,random,os

# corel5k X column order in corel5k_six_view.mat features field:
# 0 DenseHue 100, 1 DenseSift 1000, 2 Gist 512, 3 Hsv 4096, 4 Rgb 4096, 5 Lab 4096.
_EXPECTED_VIEW_DIMS = (100, 1000, 512, 4096, 4096, 4096)
_GIST_VIEW_INDEX = 2
_POWER_VIEW_INDEX = (0, 1, 3, 4, 5)

def _rank_preprocess_mode():
    mode = os.environ.get('RANK_PREPROCESS', 'r0').strip().lower()
    if mode not in ('r0', 'rgamma'):
        raise ValueError('RANK_PREPROCESS must be r0 or rgamma, got %r' % mode)
    return mode

def _l1_power_l2(v_data):
    # per-sample L1 -> x**0.25 -> per-sample L2. float64 accumulation, float32 out.
    x = np.asarray(v_data, dtype=np.float64)
    if np.any(x < 0):
        raise ValueError('power preprocess expects non-negative view values')
    l1 = np.abs(x).sum(axis=1, keepdims=True)
    np.divide(x, l1, out=x, where=l1 != 0)
    x = np.power(x, 0.25)
    l2 = np.sqrt(np.square(x).sum(axis=1, keepdims=True))
    np.divide(x, l2, out=x, where=l2 != 0)
    return x.astype(np.float32)

def _transform_views_then_mask(mv_data, inc_view_indicator):
    dims = tuple(int(v_data.shape[1]) for v_data in mv_data)
    if dims != _EXPECTED_VIEW_DIMS:
        raise ValueError('view dims %s != expected %s' % (dims, _EXPECTED_VIEW_DIMS))
    mode = _rank_preprocess_mode()
    if not getattr(_transform_views_then_mask, '_logged', False):
        print('RANK_PREPROCESS', mode, 'view_dims', dims, flush=True)
        _transform_views_then_mask._logged = True
    inc_mv_data = []
    for v, v_data in enumerate(mv_data):
        if mode == 'rgamma' and v in _POWER_VIEW_INDEX:
            transformed = _l1_power_l2(v_data)
        else:
            transformed = StandardScaler().fit_transform(v_data.astype(np.float32))
        # same transform-then-mask order as the official StandardScaler line
        inc_mv_data.append(transformed * inc_view_indicator[:, v:v+1])
    return inc_mv_data
# def loadMvSlDataFromMat(mat_path):
#     data = scipy.io.loadmat(mat_path)
#     mv_data = data['X'][0]
#     mv_data = [StandardScaler().fit_transform(v_data.astype(np.float32)) for v_data in mv_data]
#     if 'Y' in data.keys():
#         label = data['Y']
#     elif 'gt' in data.keys():
#         label = data['gt']
#     elif 'truth' in data.keys():
#         label = data['truth']
#     else :
#         raise ValueError('no label index key!!!',data.keys())
#     labels = labels.astype(np.float32)

#     if label.shape[0]!=mv_data[0].shape[0]:
#         for v in range(len(mv_data)):
#             mv_data[v] = mv_data[v].T
#             assert mv_data[v].shape[0]==label.shape[0]
#     return mv_data,label

def loadMvMlDataFromMat(mat_path):
    data = scipy.io.loadmat(mat_path)
    mv_data = data['X'][0]
    labels = data['label']
    labels = labels.astype(np.float32)
    if labels.min() == -1:
        labels = (labels + 1) * 0.5
    if labels.shape[0] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    elif labels.shape[1] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    if total_sample_num!=mv_data[0].shape[0]:
        mv_data = [v_data.T for v_data in mv_data]
    if total_sample_num!=labels.shape[0]:
        labels = labels.T
    assert mv_data[0].shape[0]==labels.shape[0]==total_sample_num
    ind00 = labels==0
    # mv_data = [np.delete(v_data, ind00,axis=0)) for v_data in mv_data]
    # labels = np.delete(labels,ind00,axis=0))

    mv_data = [StandardScaler().fit_transform(v_data.astype(np.float32)) for v_data in mv_data]
    # shuffle the data list
    random.seed(1)
    rand_index=list(range(total_sample_num))
    random.shuffle(rand_index)
    return [v_data[rand_index] for v_data in mv_data],labels[rand_index],total_sample_num

def loadMfDIMvMlDataFromMat(mat_path, fold_mat_path,fold_idx=0):
    # load multiple folds double incomplete multi-view multi-label data and labels 
    # mark sure the out dimension is n x d, where n is the number of samples
    data = scipy.io.loadmat(mat_path)
    datafold = scipy.io.loadmat(fold_mat_path)
    # multi-view data labels
    mv_data = data['X'][0]
    labels = data['label']
    labels = labels.astype(np.float32)
    if labels.min() == -1:
            labels = (labels + 1) * 0.5
    if labels.shape[0] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    elif labels.shape[1] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    if total_sample_num!=mv_data[0].shape[0]:
        mv_data = [v_data.T for v_data in mv_data]
    if total_sample_num!=labels.shape[0]:
        labels = labels.T
    assert mv_data[0].shape[0]==labels.shape[0]==total_sample_num
    


    folds_data = datafold['folds_data']
    folds_label = datafold['folds_label']
    folds_sample_index = datafold['folds_sample_index']
    # incomplete data, label and random_sample index
    inc_view_indicator = np.array(folds_data[0, fold_idx], 'int32')
    inc_label_indicator = np.array(folds_label[0, fold_idx], 'int32')  # incomplete label index
    sample_index = np.array(folds_sample_index[0, fold_idx], 'int32').reshape(-1)-1 # index start from 0
    labels,inc_view_indicator,inc_label_indicator = labels[sample_index],inc_view_indicator[sample_index],inc_label_indicator[sample_index]
    mv_data = [v_data[sample_index,:] for v,v_data in enumerate(mv_data)]

    assert inc_view_indicator.shape[0]==inc_label_indicator.shape[0]==sample_index.shape[0]==labels.shape[0]
    # incomplete data construction and normalization fill with 0 value
    # R0: StandardScaler on every view, then multiply the view mask (official line).
    # Rγ (RANK_PREPROCESS=rgamma): L1 -> x**0.25 -> L2 on views {0,1,3,4,5}; Gist stays StandardScaler.
    inc_mv_data = _transform_views_then_mask(mv_data, inc_view_indicator)
    ###### or fill it with random noise 
    # nor_mv_data = [(StandardScaler(),fit_transform(v_data.astype(np.float32))) for v,v_data in enumerate(mv_data)]
    # inc_mv_data = [np.random.randn(v_data,shape[0], v_data.shape[1]) for v_data in nor_mv_data]
    # for v, v_ data in enumerate(nor_mv_data):
    #     inc_mv_data[v][inc_view_indicator[:, v] == 1, :] = v_data[inc_view_indicator[:, v] == 1,:].copy()
    
    # incomplete label construction
    inc_labels = labels*inc_label_indicator
    # delete data with all zero label 
    ind00 = labels.sum(axis=1)==0
    # inc_mv_data = [np.delete(v_data, ind00,axis=0) for v_data in inc_mv_data]
    # mv_data = [np.delete(v_data, ind00,axis=0) for v_data in mv_data]
    # labels = np.delete(labels,ind00,axis=0)   
    # inc_labels = np.delete(inc_labels,ind00,axis=0)   
    # inc_view_indicator = np.delete(inc_view_indicator,ind00,axis=0)
    # inc_label_indicator = np.delete(inc_label_indicator,ind00,axis=0)
    # total_sample_num = labels.shape[0]

    # print('zerolabel:',(labels.sum(-1)==2).sum())
    # confirm the incomplete setting!
    # print('inc_view_indicator',inc_view_indicator[0:5,0])
    # print('inc_mv_data',inc_mv_data[0][0:5,1])
    # print('labels->inc_label_index',labels[0,0:6],inc_label_indicator[0,0:6])
    # print('inc_labels',inc_labels[0,0:6])
    # print('inc_label==label?',np.array_equal(inc_labels,labels))
    # print('inc_mv_data==mv_data?',[np.array_equal(v_data,inc_mv_data[v]) for v,v_data in enumerate(mv_data)])
    # for 

    return inc_mv_data,inc_labels,labels,inc_view_indicator,inc_label_indicator,total_sample_num
    
class ComDataset(Dataset):
    def __init__(self,mat_path,training_ratio=0.7,val_ratio=0.15,mode='train',semisup=False):
        self.mv_data, self.labels, self.total_sample_num= loadMvMlDataFromMat(mat_path)
        self.train_sample_num = math.ceil(self.total_sample_num * training_ratio)
        self.val_sample_num = math.ceil(self.total_sample_num * val_ratio)
        self.test_sample_num = self.total_sample_num - self.train_sample_num - self.val_sample_num
        if mode=='train':
            self.cur_mv_data = [v_data[:self.train_sample_num] for v_data in self.mv_data]
            self.cur_labels = self.labels[:self.train_sample_num]
        elif mode=='val':
            self.cur_mv_data = [v_data[self.train_sample_num:self.train_sample_num+self.val_sample_num] for v_data in self.mv_data]
            self.cur_labels = self.labels[self.train_sample_num:self.train_sample_num+self.val_sample_num]
        else:
            self.cur_mv_data = [v_data[self.train_sample_num+self.val_sample_num:] for v_data in self.mv_data]
            self.cur_labels = self.labels[self.train_sample_num+self.val_sample_num:]
        # print('is_train:',is_train,'num:',self.cur_mv_data[0].shape)
        self.mode = mode
        self.classes_num = self.labels.shape[1]
        self.d_list = [da.shape[1] for da in self.mv_data]
    def __len__(self):
        if self.mode == 'train':
            return self.train_sample_num 
        elif self.mode == 'val':
            return self.val_sample_num 
        else: return self.test_sample_num 
    
    def __getitem__(self, index):
        # index = index if self.is_train else self.train_sample_num+index
        data = [torch.tensor(v[index],dtype=torch.float) for v in self.cur_mv_data] 
        label = torch.tensor(self.cur_labels[index], dtype=torch.float)
        return data,label, data, label

class IncDataset(Dataset):
    def __init__(self,mat_path, fold_mat_path, training_ratio=0.7, val_ratio=0.15, fold_idx=0, mode='train',semisup=False):
        inc_mv_data, inc_labels, labels, inc_V_ind, inc_L_ind, total_sample_num= loadMfDIMvMlDataFromMat(mat_path,fold_mat_path,fold_idx)
        # inc_mv_data, inc_labels, labels, inc_V_ind, inc_L_ind, total_sample_num= loadMvMlDataFromMat(mat_path)
        self.train_sample_num = math.ceil(total_sample_num * training_ratio)
        self.val_sample_num = math.ceil(total_sample_num * val_ratio)
        self.test_sample_num = total_sample_num - self.train_sample_num - self.val_sample_num
        if mode=='train':
            self.cur_mv_data = [v_data[:self.train_sample_num] for v_data in inc_mv_data]
            self.cur_labels = inc_labels[:self.train_sample_num]
            self.cur_inc_V_ind = inc_V_ind[:self.train_sample_num]
            self.cur_inc_L_ind = inc_L_ind[:self.train_sample_num]
        elif mode=='val':
            self.cur_mv_data = [v_data[self.train_sample_num:self.train_sample_num+self.val_sample_num] for v_data in inc_mv_data]
            self.cur_labels = labels[self.train_sample_num:self.train_sample_num+self.val_sample_num]
            self.cur_inc_V_ind = inc_V_ind[self.train_sample_num:self.train_sample_num+self.val_sample_num]
            self.cur_inc_L_ind = np.ones_like(inc_L_ind[self.train_sample_num:self.train_sample_num+self.val_sample_num])
        else:
            self.cur_mv_data = [v_data[self.train_sample_num+self.val_sample_num:] for v_data in inc_mv_data]
            self.cur_labels = labels[self.train_sample_num+self.val_sample_num:]
            self.cur_inc_V_ind = inc_V_ind[self.train_sample_num+self.val_sample_num:]
            self.cur_inc_L_ind = np.ones_like(inc_L_ind[self.train_sample_num+self.val_sample_num:])

        self.mode = mode
        self.classes_num = labels.shape[1]
        self.d_list = [da.shape[1] for da in inc_mv_data]

    def __len__(self):
        if self.mode == 'train':
            return self.train_sample_num 
        elif self.mode == 'val':
            return self.val_sample_num 
        else: return self.test_sample_num 
    
    def __getitem__(self, index):
        # index = index if self.is_train else self.train_sample_num+index
        data = [torch.tensor(v[index],dtype=torch.float) for v in self.cur_mv_data] 
        label = torch.tensor(self.cur_labels[index], dtype=torch.float)
        inc_V_ind = torch.tensor(self.cur_inc_V_ind[index], dtype=torch.int32)
        inc_L_ind = torch.tensor(self.cur_inc_L_ind[index], dtype=torch.int32)
        return data,label,inc_V_ind,inc_L_ind

def getComDataloader(matdata_path,training_ratio=0.7,val_ratio=0.15,mode='train',batch_size=1,num_workers=1,shuffle=False):
    dataset = ComDataset(matdata_path, training_ratio=training_ratio, val_ratio=val_ratio, mode=mode)
    dataloder = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    return dataloder,dataset
 
def getIncDataloader(matdata_path, fold_matdata_path, training_ratio=0.7, val_ratio=0.15, fold_idx=0, mode='train',batch_size=1,num_workers=1,shuffle=False):
    dataset = IncDataset(matdata_path, fold_matdata_path, training_ratio=training_ratio, val_ratio=val_ratio, mode=mode, fold_idx=fold_idx)
    dataloder = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    return dataloder,dataset
    
if __name__=='__main__':
    # dataloder,dataset = getComDataloader('/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat',training_ratio=0.7,mode='train',batch_size=3,num_workers=2)
    dataloder,dataset = getIncDataloader('/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat','/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view_MaskRatios_0.5_LabelMaskRatio_0.5_TraindataRatio_0.7.mat',training_ratio=0.7,mode='train',batch_size=3,num_workers=2)
    print(dataset.test_sample_num)
    labels = torch.tensor(dataset.cur_labels).float()
    # total_graph = torch.matmul(labels.T,labels)
    # print(dataset.__getitem__(0)[1])
    # total_graph = torch.zeros([dataset.cur_labels.shape[1],dataset.cur_labels.shape[1]])
    # total_label = []
    # for i,(data,label,inc_V_ind,inc_L_ind) in enumerate(dataloder):
    #     label = label.float()
    #     total_label = torch.cat((total_label,label),dim=0) if len(total_label)>0 else label
    #     valid_labels_sum = torch.matmul(inc_L_ind.T.float(), inc_L_ind.float()) #[c, c]
    #     graph = torch.matmul(label.T, label) 
    #     # print(torch.matmul(label.T, label)[0,0],label_sum_cx1[0,0])
    #     total_graph += graph
    # total_graph = total_graph/(torch.diag(total_graph).unsqueeze(1))
    # np.save('c.npy',total_graph.fill_diagonal_(0).numpy())

    # loadMfDIMvMlDataFromMat('/disk1/lcl/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat','/disk1/lcl/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view_MaskRatios_0.5_LabelMaskRatio_0.5_TraindataRatio_0.7.mat',fold_idx=0)
