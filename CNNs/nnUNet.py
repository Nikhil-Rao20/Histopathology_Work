from batchgenerators.utilities.file_and_folder_operations import join
from batchgenerators.utilities.file_and_folder_operations import *
import pydoc
import dynamic_network_architectures

def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr
def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(network_class)
    # sometimes things move around, this makes it so that we can at least recover some of that
    if nw_class is None:
        warnings.warn(f'Network class {network_class} not found. Attempting to locate it within '
                      f'dynamic_network_architectures.architectures...')
        import dynamic_network_architectures
        nw_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                               network_class.split(".")[-1],
                                               'dynamic_network_architectures.architectures')
        if nw_class is not None:
            print(f'FOUND IT: {nw_class}')
        else:
            raise ImportError('Network class could not be found, please check/correct your plans file')

    if deep_supervision is not None:
        architecture_kwargs['deep_supervision'] = deep_supervision

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network


model = get_network_from_plans(
    arch_class_name="dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
    arch_kwargs={
        "n_stages": 7,
        "features_per_stage": [32, 64, 128, 256, 512, 512, 512],
        "conv_op": "torch.nn.modules.conv.Conv2d",
        "kernel_sizes": [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
        "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
        "n_blocks_per_stage": [1, 3, 4, 6, 6, 6, 6],
        "n_conv_per_stage_decoder": [1, 1, 1, 1, 1, 1],
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU",
        "nonlin_kwargs": {"inplace": True},
    },
    arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
    input_channels=3,
    output_channels=1,
    allow_init=True,
    deep_supervision=True,
)
data = torch.rand((8, 1, 256, 256))
target = torch.rand(size=(8, 1, 256, 256))
outputs = model(data) 
# this will be a 6 tensors output of shapes:
# torch.Size([8, 4, 256, 256])
# torch.Size([8, 4, 128, 128])
# torch.Size([8, 4, 64, 64])
# torch.Size([8, 4, 32, 32])
# torch.Size([8, 4, 16, 16])
# torch.Size([8, 4, 8, 8])