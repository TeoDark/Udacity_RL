import torch
import math
import ast


def to_tensor(numpy_array):
    if torch.cuda.is_available():
        variable = torch.Tensor(numpy_array).cuda()
    else:
        variable = torch.Tensor(numpy_array).cpu()
    return variable

def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.cpu().data.numpy()
    return action

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def to_tensor_long(numpy_array):
    if torch.cuda.is_available():
        variable = torch.LongTensor(numpy_array).cuda()
    else:
        variable = torch.LongTensor(numpy_array).cpu()
    return variable

def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)

def save_networks(actor, critic, z_filter_running_state, params, name, talking=False):
    params_attrs = vars(params)
    all_params = {}
    for item in params_attrs.items():
        all_params[item[0]]=item[1]

    z_filter_state_params = vars(z_filter_running_state)
    z_filter_params = {}
    for item in z_filter_state_params.items():
        z_filter_params[item[0]]=str(item[1])

    network_state = {"args_params":all_params, "z_filter_state":z_filter_params , "actor":actor.state_dict(), "critic": critic.state_dict()}

    if(talking):
        print(network_state)
    
    torch.save(network_state, str(name))
