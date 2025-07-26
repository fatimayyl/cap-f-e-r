
import os
import platform
import tensorflow as tf

from sdks.novavision.src.base.download import Download
from sdks.novavision.src.base.application import Application

weight_path = '/storage/modelFER.h5'
weight_url = 'https://drive.google.com/file/d/1JBGZc7eMPCqVLWqUQhN-20yM4kS0XXER/view?usp=sharing'
output_directory = '/storage/'


def select_device(device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f'TensorFlow Python-{platform.python_version()} tensorflow-{tf.__version__} '
    device = str(device).strip().lower().replace('gpu:', '').replace('none', '')
    cpu = device == 'cpu'

    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    physical_devices = tf.config.list_physical_devices('GPU')

    if not cpu and physical_devices:
        devices = device.split(',') if device else [str(i) for i in range(len(physical_devices))]
        n = len(devices)
        if n > 1 and batch_size > 0:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'

        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            with tf.device(f'/device:GPU:{d}'):
                device_name = tf.test.gpu_device_name()
                mem = tf.config.experimental.get_memory_info(f'GPU:{d}')
                s += f"{'' if i == 0 else space}GPU:{d} ({device_name}, {mem['total'] / (1 << 20):.0f}MiB)\n"

        tf.config.set_visible_devices([physical_devices[int(d)] for d in devices], 'GPU')
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)
        arg = f'/device:GPU:{devices[0]}'
    else:
        s += 'CPU\n'
        arg = '/device:CPU:0'

    if not newline:
        s = s.rstrip()
    return arg


def load_models(config):
    models = {}
    application = Application()
    device = select_device('0' if tf.config.list_physical_devices('GPU') else 'cpu')
    models["device"] = device
    print(config)
    config_device= application.get_param(config=config,name="ConfigDevice")
    print("DEBUG: app_param_task =", config_device)

    if not config_device:
        print("Warning: ConfigExecutor param is None or empty, using default CPU device.")

        # Model dosyası yoksa indir
        if not os.path.exists(weight_path):
            print("Model file not found, attempting to download...")
            if Download.download_from_drive(weight_url, weight_path) is not None:
                print("✅ modelFER.h5 model downloaded successfully.")
            else:
                raise RuntimeError("❌ modelFER.h5 model download failed.")

        # CPU modeli mutlaka yükle
        try:
            with tf.device("/CPU:0"):
                model = tf.keras.models.load_model(weight_path)
                models["ModelCPU"] = {"model": model}
                print("✅ modelFER.h5 loaded on CPU.")
        except Exception as e:
            raise RuntimeError(f"❌ Error loading model on CPU: {str(e)}")

        return models





    if not os.path.exists(weight_path):
        if Download.download_from_drive(weight_url, weight_path) is not None:
            print("modelFER.h5 model download successfully.")
        else:
            print("modelFER.h5 model download failed.")

    model = tf.keras.models.load_model(weight_path)

    if config_device == 'GPU' and 'GPU' in device:
        with tf.device(device):
            models["ModelGPU"] = {"model": model}
    else:
        with tf.device(device):
            models["ModelCPU"] = {"model": model}


    return models































