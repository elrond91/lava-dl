import numpy as np
from lava.lib.dl.slayer import obd
import os

if __name__ == '__main__':
    train_set = obd.dataset._PropheseeAutomotive(root='/data-raid/sshresth/data/Prophesee_1mp', 
                                                delta_t = 1,
                                                train=True, 
                                                randomize_seq= False,
                                                seq_len = 999999999999)
    
    test_set = obd.dataset._PropheseeAutomotive(root='/data-raid/sshresth/data/Prophesee_1mp', 
                                                delta_t = 1,
                                                train=False, 
                                                randomize_seq= False,
                                                seq_len = 999999999999)
                    

    out_path = '/data-raid/sshresth/data/Prophesee_fl'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    train_path = out_path + os.path.sep + '/train'
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        
    test_path = out_path + os.path.sep + '/val'
    if not os.path.exists(test_path):
        os.makedirs(test_path)
        
    print('starting...')
        
    for idx in range(len(train_set)):
        name = train_set.get_name(idx) 
        images, annotations = train_set[idx]
        if not os.path.exists(train_path + os.path.sep + name):
            os.makedirs(train_path + os.path.sep + name)
            os.makedirs(train_path + os.path.sep + name +  os.path.sep + 'events')
            os.makedirs(train_path + os.path.sep + name +  os.path.sep + 'labels')
            
        idd = 0
        for events, label in zip(images, annotations):
            np.savez_compressed(train_path + os.path.sep + name +  os.path.sep + 
                                'events' + os.path.sep + '{:05d}'.format(idd) + '.npz', a=events)
            np.savez_compressed(train_path + os.path.sep + name +  os.path.sep + 
                                    'labels' + os.path.sep + '{:05d}'.format(idd) + '.npz', a=label)
            
            events_loaded = np.load(train_path + os.path.sep + name +  os.path.sep + 
                                'events' + os.path.sep + '{:05d}'.format(idd) + '.npz')['a']

            label_loaded = np.load(train_path + os.path.sep + name +  os.path.sep + 
                                'labels' + os.path.sep + '{:05d}'.format(idd) + '.npz',
                                allow_pickle='TRUE')['a'].item()
            idd += 1
        print(idx, '/', len(train_set), ' train_set: ', name)
            
    for idx in range(len(test_set)):
        name = test_set.get_name(idx) 
        images, annotations = test_set[idx]
        if not os.path.exists(test_path + os.path.sep + name):
            os.makedirs(test_path + os.path.sep + name)
            os.makedirs(test_path + os.path.sep + name +  os.path.sep + 'events')
            os.makedirs(test_path + os.path.sep + name +  os.path.sep + 'labels')
            
        idd = 0
        for events, label in zip(images, annotations):
            np.savez_compressed(test_path + os.path.sep + name +  os.path.sep + 
                                'events' + os.path.sep + '{:05d}'.format(idd) + '.npz', a=events)
            np.savez_compressed(test_path + os.path.sep + name +  os.path.sep + 
                                    'labels' + os.path.sep + '{:05d}'.format(idd) + '.npz', a=label)
            
            events_loaded = np.load(test_path + os.path.sep + name +  os.path.sep + 
                                'events' + os.path.sep + '{:05d}'.format(idd) + '.npz')['a']

            label_loaded = np.load(test_path + os.path.sep + name +  os.path.sep + 
                                'labels' + os.path.sep + '{:05d}'.format(idd) + '.npz',
                                allow_pickle='TRUE')['a'].item()
            idd += 1
        print(idx, '/', len(test_set), ' test_set: ', name)