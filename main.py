import tensorflow as tf

from .model import ESRGAN


mode: str = "train"


def main():
    cfg = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=cfg) as sess:
        model = ESRGAN(sess=sess)
        model.summary()

        if mode == "train":
            model.train()

        if mode == "test":
            model.test()


if __name__ == "__main__":
    main()
