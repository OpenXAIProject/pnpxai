def get_default_channel_dim(modality):
    if modality == 'image':
        return 1
    elif modality == 'text':
        return -1
    elif modality == ('image', 'text'):
        return tuple(get_default_channel_dim(m) for m in modality)