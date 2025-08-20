import os
import platform
import tensorflow as tf

from sdks.novavision.src.base.download import Download
from sdks.novavision.src.base.application import Application
from capsules.FacialEmotionRecognition.src.models import PackageModel

weight_path = '/storage/video.h5'
weight_url = 'https://drive.google.com/file/d/1VsB0p2uomNRmFuyAC6eqQOR24b1c3aoY/view?usp=drive_link'


def select_device(device='', batch_size=0, newline=True):
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


def load_models():
    models = {}
    model = {}
    application = Application()

    device = select_device('0' if tf.config.list_physical_devices('GPU') else 'cpu')
    models["device"] = device

    # Burada "config" argümanını öncelikle elde etmen gerekiyor.
    # Mesela config, application veya çağıran yerden geliyor olabilir.
    # Senin durumda elinde config yoksa, önce config alman gerek.
    # Örnek olarak şöyle yapabiliriz:

    # Öncelikle "FacialEmotionRecognition" paketinin config nesnesini al:
    package_config = application.get_param(name="FacialEmotionRecognition", config="config")
    # Ya da eğer parametre sırası farklıysa:
    # package_config = application.get_param(config="FacialEmotionRecognition", name="config")

    # Şimdi ConfigDevice bilgisini al:
    device_preference = application.get_param(config=package_config, name="ConfigDevice")

    # Eğer device_preference bir config objesi ise, içinden value'yu al:
    config_device = device_preference.value if device_preference is not None else "cpu"

    # Model dosyası yoksa indir
    if not os.path.exists(weight_path):
        if Download.download_from_drive(weight_url, weight_path) is not None:
            print("modelFER.h5 model download successfully.")
        else:
            print("modelFER.h5 model download failed.")

    # Model yükleniyor
    model["model"] = tf.keras.models.load_model(weight_path)

    # Cihaza uygun model ataması
    with tf.device(device):
        if config_device == 'GPU' and 'GPU' in device:
            models["ModelGPU"] = model
        else:
            models["ModelCPU"] = model

    return models
