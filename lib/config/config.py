from yacs.config import CfgNode as CN

__all__ = ['update_config'] 

config = CN()

# configure the system related matters, such as gpus, cudnn and so on
config.SYSTEM = CN()
# config.SYSTEM.project_root = '' # not use in the project, we will use the args.project_path instead
# TODO: discrad the multigpus or num_gpus
config.SYSTEM.multigpus = False  # to determine whether use multi gpus to train or test(data parallel)
config.SYSTEM.num_gpus = 1    # decide the num_gpus and 
config.SYSTEM.gpus = [0]

config.SYSTEM.cudnn = CN()
config.SYSTEM.cudnn.benchmark = True
config.SYSTEM.cudnn.deterministic = False
config.SYSTEM.cudnn.enable = True

# about use the distributed
config.SYSTEM.distributed = CN()
config.SYSTEM.distributed.use = False
# configure the log things
config.LOG = CN()
config.LOG.log_output_dir = ''
config.LOG.tb_output_dir = '' # tensorboard log output dir
config.LOG.vis_dir = '/export/home/chengyh/Anomaly_DA/output/vis'

# configure the dataset 
config.DATASET = CN()
config.DATASET.name = ''
config.DATASET.seed = 2020
config.DATASET.read_format = 'opencv'
# config.DATASET.train_size = (256,256) # h*w
config.DATASET.train_path = ''
config.DATASET.train_clip_length = 5
config.DATASET.train_clip_step = 1
# config.DATASET.test_size = (256, 256)
config.DATASET.test_path = ''
config.DATASET.test_clip_length = 5
config.DATASET.test_clip_step = 1
config.DATASET.gt_path = ''
config.DATASET.number_of_class = 1 # use in changing the label to one hot
config.DATASET.score_normalize = True
config.DATASET.decidable_idx = 1
config.DATASET.smooth = CN()
config.DATASET.smooth.guassian = True
config.DATASET.smooth.guassian_sigma = 10
config.DATASET.mini_dataset = CN()
config.DATASET.mini_dataset.samples = 2


# ****************configure the argument of the data*************************
config.ARGUMENT = CN()
#========================Train Augment===================
config.ARGUMENT.train = CN()
config.ARGUMENT.train.use = False
config.ARGUMENT.train.resize = CN()
config.ARGUMENT.train.resize.use = False
config.ARGUMENT.train.resize.height = 32
config.ARGUMENT.train.resize.width = 32
config.ARGUMENT.train.grayscale = CN()
config.ARGUMENT.train.grayscale.use = False
config.ARGUMENT.train.flip = CN()
config.ARGUMENT.train.flip.use = False
config.ARGUMENT.train.flip.p = 0.5
config.ARGUMENT.train.rote = CN()
config.ARGUMENT.train.rote.use = False
config.ARGUMENT.train.rote.degrees = 10
#-------------------Normal------------------------
config.ARGUMENT.train.normal = CN()
config.ARGUMENT.train.normal.use = False
config.ARGUMENT.train.normal.mean = [0.485, 0.456, 0.406]
config.ARGUMENT.train.normal.std = [0.229, 0.224, 0.225]
#========================Val Augment===================
config.ARGUMENT.val = CN()
config.ARGUMENT.val.use = False
config.ARGUMENT.val.resize = CN()
config.ARGUMENT.val.resize.use = False
config.ARGUMENT.val.resize.height = 32
config.ARGUMENT.val.resize.width = 32
config.ARGUMENT.val.grayscale = CN()
config.ARGUMENT.val.grayscale.use = False
config.ARGUMENT.val.flip = CN()
config.ARGUMENT.val.flip.use = False
config.ARGUMENT.val.flip.p = 0.5
config.ARGUMENT.val.rote = CN()
config.ARGUMENT.val.rote.use = False
config.ARGUMENT.val.rote.degrees = False
#-------------------Normal------------------------
config.ARGUMENT.val.normal = CN()
config.ARGUMENT.val.normal.use = False
config.ARGUMENT.val.normal.mean = [0.485, 0.456, 0.406]
config.ARGUMENT.val.normal.std = [0.229, 0.224, 0.225]
# *************************************************************************

# configure the related video things
config.VIDEO = CN()
config.VIDEO.height = 0 
config.VIDEO.width = 0
config.VIDEO.num_frames = 0

# configure the model related things
# TODO: configure the structure of the model
config.MODEL = CN()
config.MODEL.name = ''   # the name of the network, such as resnet
config.MODEL.type = ''   # the type of the network, such as resnet50, resnet101 or resnet152
config.MODEL.eval_hooks = []  # determine the hooks use in the training
config.MODEL.hooks = []  # determine the hooks use in the training
config.MODEL.flow_model_path = ''
config.MODEL.discriminator_channels = []
config.MODEL.pretrain_model = ''
config.MODEL.detector_config = ''
config.MODEL.detector_model_path = ''

# configure the resume
config.RESUME = CN()
config.RESUME.flag = False
config.RESUME.checkpoint_path = ''

# configure the freezing layers
config.FINETUNE = CN()
config.FINETUNE.flag = False
config.FINETUNE.layer_list = []

# configure the training process
config.TRAIN = CN()
config.TRAIN.batch_size = 2
config.TRAIN.start_step = 0
config.TRAIN.max_steps = 20000  # epoch * len(dataset)
config.TRAIN.log_step = 5  # the step to print the info
config.TRAIN.mini_eval_step = 10 # the step to exec the light-weight eval
config.TRAIN.eval_step = 100 # the step to use the evaluate function
config.TRAIN.save_step = 500  # the step to save the model
config.TRAIN.epochs = 1 
config.TRAIN.loss = ['mse', 'cross']  
config.TRAIN.loss_coefficients = [0.5, 0.5] # the len must pair with the loss
#---------------Optimizer configure---------------
config.TRAIN.optimizer = CN()
config.TRAIN.optimizer.name = 'adam'
config.TRAIN.optimizer.lr = 1e-3
config.TRAIN.optimizer.momentum = 0.9
config.TRAIN.optimizer.weight_decay = 0.0001
config.TRAIN.optimizer.nesterov = False
#==================Adversarial Setting==================
config.TRAIN.optimizer.adversarial = CN()
config.TRAIN.optimizer.adversarial.g_lr = 1e-2
config.TRAIN.optimizer.adversarial.d_lr = 1e-2
#-----------------Scheduler configure--------------
config.TRAIN.scheduler = CN()
config.TRAIN.scheduler.use = True
config.TRAIN.scheduler.name = 'none'
config.TRAIN.scheduler.step_size = 30 # the numebr of the iter, should be len(dataset) * want_epochs
config.TRAIN.scheduler.gamma = 0.1
config.TRAIN.scheduler.T_max = 300 # use ine the cosine annealing LR
config.TRAIN.scheduler.eta_min = 0
#----------------Train save configure------------
config.TRAIN.split = ''
config.TRAIN.model_output = '' # use save the final model
config.TRAIN.checkpoint_output = '' # use to save the intermediate results, including lr, optimizer, state_dict...
config.TRAIN.pusedo_data_path = ''
#-------------------cluster setting--------------
config.TRAIN.cluster = CN()
config.TRAIN.cluster.k = 10

# configure the val process
config.VAL = CN()
config.VAL.name = ''
config.VAL.path = '' # if not use the data in the TRAIN.test_path
config.VAL.batch_size = 2

# configure the test process
config.TEST = CN()
config.TEST.name = ''
config.TEST.path = '' # if not use the data in the TRAIN.test_path
config.TEST.model_file = ''
config.TEST.result_output = ''
config.TEST.label_folder = ''


def _get_cfg_defaults():
    '''
    Get the config template
    NOT USE IN OTHER FILES!!
    '''
    return config.clone()


def update_config(yaml_path, opts):
    '''
    Make the template update based on the yaml file
    '''
    print('=>Merge the config with {}\t'.format(yaml_path))
    cfg = _get_cfg_defaults()
    cfg.merge_from_file(yaml_path)
    cfg.merge_from_list(opts)
    cfg.freeze()

    return cfg
    