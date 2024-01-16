from lava.lib.dl.slayer import obd
import torch


if __name__ == '__main__':
    
    delta_t = 10
    
    print('starting...')
    with open("/home/lecampos/elrond91/lava-dl/data_mean_std_" + str(delta_t) + ".txt", "a") as myfile:
        myfile.write("starting... \n")
            
    root = '/home/lecampos/data/prophesee'
    train_set = obd.dataset.PropheseeAutomotive(root=root, 
                                                train=True, 
                                                delta_t=delta_t, 
                                                seq_len=99999999999,
                                                events_ratio=0.04,
                                                #size=(720, 1280),
                                                randomize_seq=False, 
                                                augment_prob=0.0)
    
    test_set = obd.dataset.PropheseeAutomotive(root=root, 
                                            train=False, 
                                            delta_t=delta_t, 
                                            seq_len=99999999999,
                                            events_ratio=0.04,
                                            #size=(720, 1280),
                                            randomize_seq=False, 
                                            augment_prob=0.0)
    channel_means = []
    channel_stds = []
    data_idx = 0
    for image, annotation in train_set:
        for idx in range(image.shape[-1]):
            test = image[:,:,:,idx]
            mu = test.mean((1,2))
            sigma = test.std((1,2))
            channel_means.append(mu)
            channel_stds.append(sigma)
        print('computed ' + str(data_idx) + '/' + str(len(train_set)))
        print('channel_mean ' + str(torch.stack(channel_means).mean(0)) + ' channel_std ' + str(torch.stack(channel_stds).mean(0)))
        
        with open("/home/lecampos/elrond91/lava-dl/data_mean_std_" + str(delta_t) + ".txt", "a") as myfile:
            myfile.write('computed ' + str(data_idx) + '/' + str(len(train_set)) + '\n')
            myfile.write('channel_mean ' + str(torch.stack(channel_means).mean(0)) + ' channel_std ' + str(torch.stack(channel_stds).mean(0)) + '\n')
        data_idx += 1
    channel_mean = torch.stack(channel_means).mean(0)
    channel_std = torch.stack(channel_stds).mean(0)
    
    print('train')
    print(channel_mean)
    print(channel_std)
    
    with open("/home/lecampos/elrond91/lava-dl/data_mean_std_" + str(delta_t) + ".txt", "a") as myfile:
        myfile.write('train \n')
        myfile.write('channel_mean ' + str(channel_mean[0]) + ' ' + str(channel_mean[1]) + '\n')
        myfile.write('channel_std ' + str(channel_std[0]) + ' ' + str(channel_std[1]) + '\n')
    
    channel_means = []
    channel_stds = []
    data_idx = 0
    for image, annotation in test_set:
        for idx in range(image.shape[-1]):
            test = image[:,:,:,idx]
            mu = test.mean((1,2))
            sigma = test.std((1,2))
            channel_means.append(mu)
            channel_stds.append(sigma)
        print(' computed ' + str(data_idx) + '/' + str(len(test_set)))
        print('channel_mean ' + str(torch.stack(channel_means).mean(0)) + ' channel_std ' + str(torch.stack(channel_stds).mean(0)))
        with open("/home/lecampos/elrond91/lava-dl/data_mean_std_" + str(delta_t) + ".txt", "a") as myfile:
            myfile.write('computed ' + str(data_idx) + '/' + str(len(test_set)) + '\n')
            myfile.write('channel_mean ' + str(torch.stack(channel_means).mean(0)) + ' channel_std ' + str(torch.stack(channel_stds).mean(0)) + '\n')
        data_idx += 1

    channel_mean = torch.stack(channel_means).mean(0)
    channel_std = torch.stack(channel_stds).mean(0) 
    
    print('test')
    print(channel_mean)
    print(channel_std)
    
    with open("/home/lecampos/elrond91/lava-dl/data_mean_std_" + str(delta_t) + ".txt", "a") as myfile:
        myfile.write('test \n')
        myfile.write('channel_mean ' + str(channel_mean[0]) + ' ' + str(channel_mean[1]) + '\n')
        myfile.write('channel_std ' + str(channel_std[0]) + ' ' + str(channel_std[1]) + '\n')