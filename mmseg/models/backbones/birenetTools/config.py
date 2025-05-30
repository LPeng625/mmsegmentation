
from .ops import createConvFunc

nets = {
    'baseline': {
        'layer0':  'cv',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'cv',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'cv',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'c-v15': {
        'layer0':  'cd',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'cv',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'cv',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'a-v15': {
        'layer0':  'ad',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'cv',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'cv',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'r-v15': {
        'layer0':  'rd',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'cv',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'cv',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'cvvv4': {
        'layer0':  'cd',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'cd',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'cd',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cd',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'avvv4': {
        'layer0':  'ad',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'ad',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'ad',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'ad',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'rvvv4': {
        'layer0':  'rd',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'rd',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'rd',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'rd',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'cccv4': {
        'layer0':  'cd',
        'layer1':  'cd',
        'layer2':  'cd',
        'layer3':  'cv',
        'layer4':  'cd',
        'layer5':  'cd',
        'layer6':  'cd',
        'layer7':  'cv',
        'layer8':  'cd',
        'layer9':  'cd',
        'layer10': 'cd',
        'layer11': 'cv',
        'layer12': 'cd',
        'layer13': 'cd',
        'layer14': 'cd',
        'layer15': 'cv',
        },
    'aaav4': {
        'layer0':  'ad',
        'layer1':  'ad',
        'layer2':  'ad',
        'layer3':  'cv',
        'layer4':  'ad',
        'layer5':  'ad',
        'layer6':  'ad',
        'layer7':  'cv',
        'layer8':  'ad',
        'layer9':  'ad',
        'layer10': 'ad',
        'layer11': 'cv',
        'layer12': 'ad',
        'layer13': 'ad',
        'layer14': 'ad',
        'layer15': 'cv',
        },
    'rrrv4': {
        'layer0':  'rd',
        'layer1':  'rd',
        'layer2':  'rd',
        'layer3':  'cv',
        'layer4':  'rd',
        'layer5':  'rd',
        'layer6':  'rd',
        'layer7':  'cv',
        'layer8':  'rd',
        'layer9':  'rd',
        'layer10': 'rd',
        'layer11': 'cv',
        'layer12': 'rd',
        'layer13': 'rd',
        'layer14': 'rd',
        'layer15': 'cv',
        },
    'c16': {
        'layer0':  'cd',
        'layer1':  'cd',
        'layer2':  'cd',
        'layer3':  'cd',
        'layer4':  'cd',
        'layer5':  'cd',
        'layer6':  'cd',
        'layer7':  'cd',
        'layer8':  'cd',
        'layer9':  'cd',
        'layer10': 'cd',
        'layer11': 'cd',
        'layer12': 'cd',
        'layer13': 'cd',
        'layer14': 'cd',
        'layer15': 'cd',
        },
    'a16': {
        'layer0':  'ad',
        'layer1':  'ad',
        'layer2':  'ad',
        'layer3':  'ad',
        'layer4':  'ad',
        'layer5':  'ad',
        'layer6':  'ad',
        'layer7':  'ad',
        'layer8':  'ad',
        'layer9':  'ad',
        'layer10': 'ad',
        'layer11': 'ad',
        'layer12': 'ad',
        'layer13': 'ad',
        'layer14': 'ad',
        'layer15': 'ad',
        },
    'r16': {
        'layer0':  'rd',
        'layer1':  'rd',
        'layer2':  'rd',
        'layer3':  'rd',
        'layer4':  'rd',
        'layer5':  'rd',
        'layer6':  'rd',
        'layer7':  'rd',
        'layer8':  'rd',
        'layer9':  'rd',
        'layer10': 'rd',
        'layer11': 'rd',
        'layer12': 'rd',
        'layer13': 'rd',
        'layer14': 'rd',
        'layer15': 'rd',
        },
    # 'carv4': {
    #     'layer0':  'cd',
    #     'layer1':  'ad',
    #     'layer2':  'rd',
    #     'layer3':  'cv',
    #     'layer4':  'cd',
    #     'layer5':  'ad',
    #     'layer6':  'rd',
    #     'layer7':  'cv',
    #     'layer8':  'cd',
    #     'layer9':  'ad',
    #     'layer10': 'rd',
    #     'layer11': 'cv',
    #     'layer12': 'cd',
    #     'layer13': 'ad',
    #     'layer14': 'rd',
    #     'layer15': 'cv',
    #     },
    'carv4': {
        'layer0': 'cv',
        'layer1': 'cd',
        'layer2': 'ad',
        'layer3': 'rd',
        'layer4': 'cv',
        'layer5': 'cd',
        'layer6': 'ad',
        'layer7': 'rd',
        'layer8': 'cv',
        'layer9': 'cd',
        'layer10': 'ad',
        'layer11': 'rd',
        'layer12': 'cv',
        'layer13': 'cd',
        'layer14': 'ad',
        'layer15': 'rd',
    },
    }


def config_model(model):
    model_options = list(nets.keys())
    assert model in model_options, \
        'unrecognized model, please choose from %s' % str(model_options)

    print(str(nets[model]))

    pdcs = []
    for i in range(16):
        layer_name = 'layer%d' % i
        op = nets[model][layer_name]
        pdcs.append(createConvFunc(op))

    return pdcs

def config_model_converted(model):
    model_options = list(nets.keys())
    assert model in model_options, \
        'unrecognized model, please choose from %s' % str(model_options)

    print(str(nets[model]))

    pdcs = []
    for i in range(16):
        layer_name = 'layer%d' % i
        op = nets[model][layer_name]
        pdcs.append(op)

    return pdcs

