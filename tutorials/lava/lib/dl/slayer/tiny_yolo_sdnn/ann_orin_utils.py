

def rename_single_head_model(model):
    model._modules['layers_back_bone']._modules['0_convbatch']._modules['conv_0'] = \
    model._modules['layers_back_bone']._modules['0_convbatch']._modules.pop('conv')
    model._modules['layers_back_bone']._modules['0_convbatch']._modules['bn_0'] = \
    model._modules['layers_back_bone']._modules['0_convbatch']._modules.pop('bn')

    model._modules['layers_back_bone']._modules['2_convbatch']._modules['conv_2'] = \
    model._modules['layers_back_bone']._modules['2_convbatch']._modules.pop('conv')
    model._modules['layers_back_bone']._modules['2_convbatch']._modules['bn_2'] = \
    model._modules['layers_back_bone']._modules['2_convbatch']._modules.pop('bn')

    model._modules['layers_back_bone']._modules['4_convbatch']._modules['conv_4'] = \
    model._modules['layers_back_bone']._modules['4_convbatch']._modules.pop('conv')
    model._modules['layers_back_bone']._modules['4_convbatch']._modules['bn_4'] = \
    model._modules['layers_back_bone']._modules['4_convbatch']._modules.pop('bn')   
    
    model._modules['layers_back_bone']._modules['6_convbatch']._modules['conv_6'] = \
    model._modules['layers_back_bone']._modules['6_convbatch']._modules.pop('conv')
    model._modules['layers_back_bone']._modules['6_convbatch']._modules['bn_6'] = \
    model._modules['layers_back_bone']._modules['6_convbatch']._modules.pop('bn')  
    
    model._modules['layers_back_bone']._modules['8_convbatch']._modules['conv_8'] = \
    model._modules['layers_back_bone']._modules['8_convbatch']._modules.pop('conv')
    model._modules['layers_back_bone']._modules['8_convbatch']._modules['bn_8'] = \
    model._modules['layers_back_bone']._modules['8_convbatch']._modules.pop('bn')  

    model._modules['layers_back_bone']._modules['10_convbatch']._modules['conv_10'] = \
    model._modules['layers_back_bone']._modules['10_convbatch']._modules.pop('conv')
    model._modules['layers_back_bone']._modules['10_convbatch']._modules['bn_10'] = \
    model._modules['layers_back_bone']._modules['10_convbatch']._modules.pop('bn') 
    
    model._modules['layers_back_bone']._modules['12_convbatch']._modules['conv_12'] = \
    model._modules['layers_back_bone']._modules['12_convbatch']._modules.pop('conv')
    model._modules['layers_back_bone']._modules['12_convbatch']._modules['bn_12'] = \
    model._modules['layers_back_bone']._modules['12_convbatch']._modules.pop('bn') 
 
    model._modules['layers_back_bone']._modules['13_convbatch']._modules['conv_13'] = \
    model._modules['layers_back_bone']._modules['13_convbatch']._modules.pop('conv')
    model._modules['layers_back_bone']._modules['13_convbatch']._modules['bn_13'] = \
    model._modules['layers_back_bone']._modules['13_convbatch']._modules.pop('bn') 
    
    return model
    