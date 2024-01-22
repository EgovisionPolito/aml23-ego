import glob
from abc import ABC
import pandas as pd
from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger
import numpy as np

class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        split: str (D1, D2 or D3)
        modalities: list(str, str, ...)
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(self.list_file)
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat

        if self.load_feat:
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                model_features = pd.DataFrame(pd.read_pickle(os.path.join("saved_features",
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                          pickle_name))['features'])[["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")

    def _get_train_indices(self, record, modality='RGB'):
        ##################################################################
        # TODO: implement sampling for training mode                     #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################
        raise NotImplementedError("You should implement _get_train_indices")

    def _get_val_indices(self, record, modality):
        ##################################################################
        # TODO: implement sampling for testing mode                      #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################

        record_num_frames = record.num_frames[modality]
        num_frames_per_clip = self.num_frames_per_clip[modality]
        desired_num_frames = num_frames_per_clip * self.num_clips
        sampled_frames_inidices_list = []

        ##* DENSE Sampling
        # frames_interval = 2
        # clip_radius = (num_frames_per_clip // 2) * frames_interval
        # if record_num_frames > desired_num_frames:
        #     for clip_number in range(self.num_clips):
        #             clip_central_point = np.random.randint(clip_radius, record_num_frames-clip_radius+2) # se record_num_frames=80 e cp=64 => sampled_frames_inidices_list=[48,..,64,..78], per questo il +2
        #             clip_frames_inidices_list = list(range(clip_central_point-clip_radius, clip_central_point+clip_radius, frames_interval))
        #             #*Se volessimo una lista di array numpy dove ogni array ha i frame di una clip
        #             #clip_frames_indices_list = np.arange(clip_central_point-clip_radius, clip_central_point+clip_radius, frames_interval)
        #             #sampled_frames_inidices_list.append(clip_frames_indices_list)
        #             #*Caso di una lista piatta con solo indici
        #             sampled_frames_inidices_list.extend(clip_frames_inidices_list)
        # else: #TODO: se il record ha troppi pochi frame non so bene come gestire la cosa, per ora prendo tutti quelli del record e duplico quelli che mancano... Si può migliorare
        #     offset = desired_num_frames - record_num_frames
        #     clip_frames_inidices_list = list(index for index in range(0, record_num_frames))
        #     clip_frames_inidices_list.extend(index for index in range(0, offset))
        #     sampled_frames_inidices_list.extend(clip_frames_inidices_list)

        # if(len(sampled_frames_inidices_list) < desired_num_frames):
        #     raise SystemError(f"For the record {record.untrimmed_video_name}, the number of extracted frames is {len(sampled_frames_inidices_list)}, that is less than the desired {desired_num_frames} frames!")
        # elif(len(sampled_frames_inidices_list) > desired_num_frames):
        #     raise SystemError(f"For the record {record.untrimmed_video_name}, the number of extracted frames is {len(sampled_frames_inidices_list)}, that is more than the desired {desired_num_frames} frames!")
        # else:
        #     return sampled_frames_inidices_list



        ##* UNIFORM Sampling
        if record_num_frames > desired_num_frames:  #if record_num_frames=300
            clips_interval = record_num_frames//self.num_clips #300/5=60 #arrotondo per difetto
            frames_interval = round(clips_interval/num_frames_per_clip) #60/16=4 #per eccesso
            for clip_number in range(self.num_clips):   #clip_number va da 0 a 4 inclusi
                    start_index = clip_number * clips_interval #0, 60, 120, 180, 240
                    end_index = (clip_number + 1) * clips_interval   #60, 120, 180, 240, 300
                    clip_frames_inidices_list = np.arange(start_index, end_index, frames_interval)   #[0,4,8,..,56,60], [60,..,120], [120,..,180], [180,..,240], [240,..,300]
                    sampled_frames_inidices_list.extend(clip_frames_inidices_list[:16])
                    logger.info(f"{record.untrimmed_video_name} {record.uid} - clip: {clip_number}, sampled_frames_inidices_list: {sampled_frames_inidices_list}")

        else: #TODO: se il record ha troppi pochi frame non so bene come gestire la cosa, per ora prendo tutti quelli del record e duplico quelli che mancano... Si può migliorare
            offset = desired_num_frames - record_num_frames
            clip_frames_inidices_list = list(index for index in range(0, record_num_frames))
            clip_frames_inidices_list.extend(index for index in range(0, offset))
            sampled_frames_inidices_list.extend(clip_frames_inidices_list)







        if(len(sampled_frames_inidices_list) < desired_num_frames):
            logger.info(f"{record.untrimmed_video_name} - record_num_frames: {record_num_frames}, clips_interval: {clips_interval}, frames_interval: {frames_interval}, frames: {sampled_frames_inidices_list}")
            raise SystemError(f"For the record {record.untrimmed_video_name} {record.uid}, the number of extracted frames is {len(sampled_frames_inidices_list)}, that is less than the desired {desired_num_frames} frames!")
        elif(len(sampled_frames_inidices_list) > desired_num_frames):
            logger.info(f"{record.untrimmed_video_name} - record_num_frames: {record_num_frames}, clips_interval: {clips_interval}, frames_interval: {frames_interval}, frames: {sampled_frames_inidices_list}")
            raise SystemError(f"For the record {record.untrimmed_video_name} {record.uid}, the number of extracted frames is {len(sampled_frames_inidices_list)}, that is more than the desired {desired_num_frames} frames!")
        else:
            return sampled_frames_inidices_list

        # sequence_len = self.num_frames_per_clip[modality] * self.num_clips # 16 * 5 in RGB = 80
        # if record.num_frames[modality] < sequence_len:                     # se lunghezza tot record (eg 60) < sequence_len (=80)
        #     offset = sequence_len - record.num_frames[modality]            # 80 - 60 = 20
        #     sequence_len = record.num_frames[modality]                     # =60
        #     sequence = list(index for index in range(0, sequence_len))     # [0,.., 59]
        #     #se il record ha meno di 80 frame (RGB) metto #offset frame duplicati
        #     sequence.extend(index for index in range(0, offset))           # appende i numeri da 0 a offest-1 (20) -> [0,.., 59, 0,..,19]
        # else:
        #     sequence = list(index for index in range(0, sequence_len))     # [0,..,79]
        
        # return sequence

    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]

        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = sample_row["features_" + m].values[0]
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p)
            images.extend(frame)
        # finally, all the transformations are applied
        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added

            idx_untrimmed = record.start_frame + idx    #start_frame decrementa di 1
            #logger.info(str(record.start_frame) + " - " + str(idx) + " - " + str(idx_untrimmed))
            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)
