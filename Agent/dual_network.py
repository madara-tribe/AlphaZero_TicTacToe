# ====================
# デュアルネットワークの作成
# ====================

from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os


DN_FILTERS  = 256 # 畳み込み層のカーネル数
DN_RESIDUAL_NUM =  19 # 残差ブロックの数
DN_INPUT_SHAPE = (3, 3, 2) # 入力シェイプ
DN_OUTPUT_SIZE = 9 # 行動数(配置先(3*3))


def conv(filters):
    return Conv2D(filters, 3, padding='same', use_bias=False,
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))

def residual_block():
    def f(x):
        sc = x
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Add()([x, sc])
        x = Activation('relu')(x)
        return x
    return f


def dual_network():
    # モデル作成済みの場合は無処理
    if os.path.exists('./model/best.h5'):
        return

    input = Input(shape=DN_INPUT_SHAPE)
    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 残差ブロック x 19
    for i in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)

    x = GlobalAveragePooling2D()(x)
    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005),
                activation='softmax', name='pi')(x)

    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation('tanh', name='v')(v)
    model = Model(inputs=input, outputs=[p,v])

    os.makedirs('./model/', exist_ok=True) # フォルダがない時は生成
    model.save('./model/best.h5') # ベストプレイヤーのモデル

    # モデルの破棄
    K.clear_session()
    del model

if __name__ == '__main__':
    dual_network()
