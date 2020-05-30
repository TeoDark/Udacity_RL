import torch
import math


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

def save_networks(actor, critic, params, name, talking=False):

    attrs = vars(params)
    all_params = ', '.join("%s: %s" % item for item in attrs.items())
    network_state = {"args_params":all_params, "actor":actor.state_dict(), "critic": critic.state_dict()}
    if(talking):
        print(network_state)
    torch.save(network_state, str(name))