from torch.utils.data import DataLoader

from data.datasets.jigsawdataloader import JigsawDataloader


def get_dataloader(data_config, mode):
    # get the iterator object
    if data_config.name == 'jigsaw_dataset':
        dataset = JigsawDataloader(data_config.params, mode)
    else:
        raise ValueError(f'Wrong dataset name: {data_config.name}')

    # batch size 1 for validation
    batch_size = int(data_config.params.batch_size) if 'train' in mode else 1

    # make the torch dataloader object
    loader = DataLoader(dataset,  # todo
                        batch_size=batch_size,
                        num_workers=data_config.params.workers,
                        drop_last=True,
                        shuffle='train' in mode,
                        pin_memory=data_config.params.load_into_memory)

    return loader
