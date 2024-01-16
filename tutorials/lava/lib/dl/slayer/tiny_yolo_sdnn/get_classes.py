from lava.lib.dl.slayer import obd
import torch
import numpy as np

if __name__ == '__main__':
    
    delta_t = 10
    file_name = 'classes'
    print('starting...')
    with open("/home/lecampos/elrond91/lava-dl/" + file_name + "_" + str(delta_t) + ".txt", "a") as myfile:
        myfile.write("starting... \n")
            
    root = '/home/lecampos/data/prophesee'
    train_set = obd.dataset.PropheseeAutomotive(root=root, 
                                                train=True, 
                                                delta_t=delta_t, 
                                                seq_len=9999999999,
                                                events_ratio=0.04,
                                                #size=(720, 1280),
                                                randomize_seq=False, 
                                                augment_prob=0.0)
    
    test_set = obd.dataset.PropheseeAutomotive(root=root, 
                                            train=False, 
                                            delta_t=delta_t, 
                                            seq_len=9999999999,
                                            events_ratio=0.04,
                                            #size=(720, 1280),
                                            randomize_seq=False, 
                                            augment_prob=0.0)
    obj_distribution = np.zeros(len(train_set.classes))
    data_idx = 0
    for image, annotations in train_set:
        for annotation in annotations:
            objects = annotation['annotation']['object']
            for obj in objects:
                obj_distribution[obj['id']] += 1
        print('computed ' + str(data_idx) + '/' + str(len(train_set)))
        print(obj_distribution)
        
        with open("/home/lecampos/elrond91/lava-dl/" + file_name + "_" + str(delta_t) + ".txt", "a") as myfile:
            myfile.write('computed ' + str(data_idx) + '/' + str(len(train_set)) + '\n')
            myfile.write(str(obj_distribution) + '\n')
        data_idx += 1
    print('final train')
    print(obj_distribution)
    
    with open("/home/lecampos/elrond91/lava-dl/" + file_name + "_" + str(delta_t) + ".txt", "a") as myfile:
        myfile.write('final train \n')
        myfile.write(str(obj_distribution) + '\n')

    
    obj_distribution = np.zeros(len(test_set.classes))
    data_idx = 0
    for image, annotation in test_set:
        for annotation in annotations:
            objects = annotation['annotation']['object']
            for obj in objects:
                obj_distribution[obj['id']] += 1
        print('computed ' + str(data_idx) + '/' + str(len(test_set)))
        print(obj_distribution)
        
        with open("/home/lecampos/elrond91/lava-dl/" + file_name + "_" + str(delta_t) + ".txt", "a") as myfile:
            myfile.write('computed ' + str(data_idx) + '/' + str(len(test_set)) + '\n')
            myfile.write(str(obj_distribution) + '\n')
        data_idx += 1


    
    print('final test')
    print(obj_distribution)
    
    with open("/home/lecampos/elrond91/lava-dl/" + file_name + "_" + str(delta_t) + ".txt", "a") as myfile:
        myfile.write('final test \n')
        myfile.write(str(obj_distribution) + '\n')