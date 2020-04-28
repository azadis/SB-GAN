def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    
    
    print('Total number of parameters: %d' % num_params)

