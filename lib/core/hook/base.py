import os
import pickle
import cv2
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tsnecuda import TSNE

from .abstract import HookBase
from lib.datatools.evaluate.utils import psnr_error
from lib.datatools.evaluate import eval_api
from lib.core.utils import tsne_vis
HOOKS = ['VisScoreHook', 'TSNEHook']

class VisScoreHook(HookBase):
    def after_step(self, current_step):
        writer = self.trainer.kwargs['writer_dict']['writer']
        global_steps = self.trainer.kwargs['writer_dict']['global_steps_{}'.format(self.trainer.kwargs['model_type'])]

        if not os.path.exists(self.trainer.config.LOG.vis_dir):
            os.mkdir(self.trainer.config.LOG.vis_dir)
        
        if current_step % self.trainer.config.TRAIN.eval_step == 0 and current_step != 0:
            result_path = os.path.join(self.trainer.config.TEST.result_output, f'{self.trainer.verbose}_cfg#{self.trainer.config_name}#step{current_step}@{self.trainer.kwargs["time_stamp"]}_results.pkl')
            with open(result_path, 'rb') as reader:
                results = pickle.load(reader)
            
            psnrs = results['psnr']
            scores = results['score']
            import ipdb; ipdb.set_trace()
            assert len(psnrs) == len(scores), 'the number of psnr and score is not equal'
            
            # plt the figure
            vis_paths = []
            for video_id in range(len(psnrs)):
                vis_path = os.path.join(self.trainer.config.LOG.vis_dir, f'{self.trainer.verbose}_cfg#{self.trainer.config_name}#step{current_step}@{self.trainer.kwargs["time_stamp"]}_vis#{video_id}.jpg')
                vis_paths.append(vis_path)
                plt.subplot(2,1,1)
                plt.plot([i for i in range(len(psnrs[video_id]))], psnrs[video_id])
                plt.ylabel('psnr')
                plt.subplot(2,1,2)
                plt.plot([i for i in range(len(scores[video_id]))], scores[video_id])
                plt.ylabel('score')
                plt.xlabel('frame')
                plt.savefig(vis_path)
            
            for vp in vis_paths:
                image = cv2.imread(vp)
                image = image[:,:,[2,1,0]]
                writer.add_image(str(vp), image, global_step=global_steps)
        
            self.trainer.logger.info(f'^^^^Finish vis @{current_step}')

class TSNEHook(HookBase):
    def after_step(self, current_step):
        writer = self.trainer.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]

        if not os.path.exists(self.trainer.config.LOG.vis_dir):
            os.mkdir(self.trainer.config.LOG.vis_dir)
        
        if current_step % self.trainer.config.TRAIN.eval_step == 0:
            vis_path = os.path.join(self.trainer.config.LOG.vis_dir, f'{self.trainer.config.DATASET.name}_tsne_model:{self.trainer.config.MODEL.name}_step:{current_step}.jpg')
            feature, feature_labels = self.trainer.analyze_feature
            tsne_vis(feature, feature_labels, vis_path)
            image = cv2.imread(vis_path)
            image = image[:,:,[2,1,0]]
            writer.add_image(str(vis_path), image, global_step=global_steps)


    
### not finish ###
class EvaluateHook(HookBase):
    def evaluate(self, current_step):
        '''
        Evaluate the results of the model
        !!! Will change, e.g. accuracy, mAP.....
        !!! Or can call other methods written by the official
        '''
        
        # video_dirs = os.listdir(self.testing_data_folder)
        # video_dirs.sort()

        # num_videos = len(video_dirs)
        # time_stamp = time.time()
        frame_num = self.config.DATASET.test_clip_length
        psnr_records=[]
        total = 0
        # for dird in video_dirs:
        for video_name in self.test_dataset_keys:
            # _temp_test_folder = os.path.join(self.testing_data_folder, dir)

            # need to improve
            # dataset = AvenueTestOld(_temp_test_folder, clip_length=frame_num)
            dataset = self.test_dataset_dict[video_name]
            len_dataset = dataset.pics_len
            test_iters = len_dataset - frame_num + 1
            test_counter = 0

            data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
            # import ipdb; ipdb.set_trace()
            psnrs = np.empty(shape=(len_dataset,),dtype=np.float32)
            # for test_input, _ in data_loader:
            for test_input in data_loader:
                test_target = test_input[:, -1].cuda()
                test_input = test_input[:, :-1].transpose_(1,2).cuda()

                g_output = self.G(test_input, test_target)
                test_psnr = psnr_error(g_output, test_target)
                test_psnr = test_psnr.tolist()
                psnrs[test_counter+frame_num-1]=test_psnr

                test_counter += 1
                total+=1
                if test_counter >= test_iters:
                    psnrs[:frame_num-1]=psnrs[frame_num-1]
                    psnr_records.append(psnrs)
                    print(f'finish test video set {video_name}')
                    break

        result_dict = {'dataset': self.config.DATASET.name, 'psnr': psnr_records, 'flow': [], 'names': [], 'diff_mask': []}
        # result_path = os.path.join(self.config.TEST.result_output, os.path.split(self.model_path)[-1])
        result_path = os.path.join(self.config.TEST.result_output, f'{self.verbose}_cfg#{self.config_name}@{self.kwargs["time_stamp"]}_results.pkl')
        with open(result_path, 'wb') as writer:
            pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)
        
        results = eval_api.evaluate('compute_auc', result_path, self.config)
        self.logger.info(results)
        return results.auc

def get_base_hooks(name):
    if name in HOOKS:
        t = eval(name)()
    else:
        raise Exception('The hook is not in amc_hooks')
    return t

        
