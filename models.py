"""
Model Architectures Module.

Contains:
- Lightweight_Proposed (original + improved with SE block option)
- MobileNetV3-Small (transfer learning with gradual unfreezing)
- ResNet50 (transfer learning with gradual unfreezing)
- Ablation study model variants
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

import config


# ============================================================
# SQUEEZE-AND-EXCITATION BLOCK
# ============================================================
def se_block(input_tensor, ratio=4):
    """
    Squeeze-and-Excitation block for channel attention.
    Lightweight version suitable for small models.
    """
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(channels // ratio, activation="relu",
                      kernel_regularizer=l2(config.L2_WEIGHT))(se)
    se = layers.Dense(channels, activation="sigmoid",
                      kernel_regularizer=l2(config.L2_WEIGHT))(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.Multiply()([input_tensor, se])


# ============================================================
# PROPOSED LIGHTWEIGHT MODEL
# ============================================================
def create_proposed_model(
    filters=None,
    dropout_mode="progressive",
    dropout_rate=0.5,
    use_bn=True,
    use_gap=True,
    use_se=False,
    name="Lightweight_Proposed",
):
    """
    Create the proposed lightweight CNN model.

    Args:
        filters: List of 3 filter sizes [f1, f2, f3]. Default: [16, 32, 64]
        dropout_mode: "progressive", "fixed", or "none"
        dropout_rate: Used when dropout_mode is "fixed"
        use_bn: Whether to use Batch Normalization
        use_gap: Whether to use Global Average Pooling (vs Flatten)
        use_se: Whether to add SE blocks after each conv block
        name: Model name
    """
    if filters is None:
        filters = [32, 64, 128]  # wider than before (was 16,32,64)

    f1, f2, f3 = filters

    # Dropout rates
    if dropout_mode == "progressive":
        dr = [config.DROPOUT_RATES["conv1"],
              config.DROPOUT_RATES["conv2"],
              config.DROPOUT_RATES["conv3"],
              config.DROPOUT_RATES["dense1"],
              config.DROPOUT_RATES["dense2"]]
    elif dropout_mode == "fixed":
        dr = [dropout_rate] * 5
    else:  # none
        dr = [0.0] * 5

    inp = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3), name="input_layer")

    # === CONV BLOCK 1 ===
    x = layers.Conv2D(f1, 3, padding="same", kernel_regularizer=l2(config.L2_WEIGHT), name="conv1_1")(inp)
    if use_bn:
        x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(f1, 3, padding="same", kernel_regularizer=l2(config.L2_WEIGHT), name="conv1_2")(x)
    if use_bn:
        x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    if use_se:
        x = se_block(x, ratio=max(2, f1 // 4))
    x = layers.MaxPooling2D(2, name="pool1")(x)
    if dr[0] > 0:
        x = layers.Dropout(dr[0], name="dropout1")(x)

    # === CONV BLOCK 2 ===
    x = layers.Conv2D(f2, 3, padding="same", kernel_regularizer=l2(config.L2_WEIGHT), name="conv2_1")(x)
    if use_bn:
        x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(f2, 3, padding="same", kernel_regularizer=l2(config.L2_WEIGHT), name="conv2_2")(x)
    if use_bn:
        x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    if use_se:
        x = se_block(x, ratio=max(2, f2 // 4))
    x = layers.MaxPooling2D(2, name="pool2")(x)
    if dr[1] > 0:
        x = layers.Dropout(dr[1], name="dropout2")(x)

    # === CONV BLOCK 3 ===
    x = layers.Conv2D(f3, 3, padding="same", kernel_regularizer=l2(config.L2_WEIGHT), name="conv3_1")(x)
    if use_bn:
        x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.Conv2D(f3, 3, padding="same", kernel_regularizer=l2(config.L2_WEIGHT), name="conv3_2")(x)
    if use_bn:
        x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)
    if use_se:
        x = se_block(x, ratio=max(2, f3 // 4))
    x = layers.MaxPooling2D(2, name="pool3")(x)
    if dr[2] > 0:
        x = layers.Dropout(dr[2], name="dropout3")(x)

    # === CLASSIFICATION HEAD ===
    if use_gap:
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    else:
        x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(128, kernel_regularizer=l2(config.L2_WEIGHT), name="dense1")(x)
    if use_bn:
        x = layers.BatchNormalization(name="bn_dense1")(x)
    x = layers.Activation("relu", name="relu_dense1")(x)
    if dr[3] > 0:
        x = layers.Dropout(dr[3], name="dropout_dense1")(x)

    x = layers.Dense(64, kernel_regularizer=l2(config.L2_WEIGHT), name="dense2")(x)
    if use_bn:
        x = layers.BatchNormalization(name="bn_dense2")(x)
    x = layers.Activation("relu", name="relu_dense2")(x)
    if dr[4] > 0:
        x = layers.Dropout(dr[4], name="dropout_dense2")(x)

    out = layers.Dense(config.NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)

    model = Model(inputs=inp, outputs=out, name=name)
    return model


# ============================================================
# IMPROVED PROPOSED MODEL (V2)
# Depthwise Separable Conv + SE Attention + Residual Connections
# ============================================================
def _dwsconv_block(x, filters, block_name, dropout_rate=0.0, use_se=True, use_residual=True):
    """
    Depthwise Separable Convolution block with optional SE and residual.

    Architecture per block:
      DepthwiseConv2D → BN → ReLU →
      Conv2D (1x1 pointwise) → BN → ReLU →
      DepthwiseConv2D → BN → ReLU →
      Conv2D (1x1 pointwise) → BN →
      [+ SE block] → [+ residual] → ReLU →
      MaxPool → Dropout
    """
    shortcut = x
    in_channels = x.shape[-1]

    # First depthwise separable conv
    x = layers.DepthwiseConv2D(
        3, padding="same", depthwise_regularizer=l2(config.L2_WEIGHT),
        name=f"{block_name}_dw1"
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn_dw1")(x)
    x = layers.Activation("relu", name=f"{block_name}_relu_dw1")(x)

    x = layers.Conv2D(
        filters, 1, padding="same", kernel_regularizer=l2(config.L2_WEIGHT),
        name=f"{block_name}_pw1"
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn_pw1")(x)
    x = layers.Activation("relu", name=f"{block_name}_relu_pw1")(x)

    # Second depthwise separable conv
    x = layers.DepthwiseConv2D(
        3, padding="same", depthwise_regularizer=l2(config.L2_WEIGHT),
        name=f"{block_name}_dw2"
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn_dw2")(x)
    x = layers.Activation("relu", name=f"{block_name}_relu_dw2")(x)

    x = layers.Conv2D(
        filters, 1, padding="same", kernel_regularizer=l2(config.L2_WEIGHT),
        name=f"{block_name}_pw2"
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn_pw2")(x)

    # SE attention
    if use_se:
        x = se_block(x, ratio=max(2, filters // 8))

    # Residual connection (with 1x1 conv if channels change)
    if use_residual:
        if in_channels != filters:
            shortcut = layers.Conv2D(
                filters, 1, padding="same",
                kernel_regularizer=l2(config.L2_WEIGHT),
                name=f"{block_name}_shortcut"
            )(shortcut)
            shortcut = layers.BatchNormalization(name=f"{block_name}_bn_shortcut")(shortcut)
        x = layers.Add(name=f"{block_name}_add")([x, shortcut])

    x = layers.Activation("relu", name=f"{block_name}_relu_out")(x)
    x = layers.MaxPooling2D(2, name=f"{block_name}_pool")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=f"{block_name}_dropout")(x)

    return x


def create_proposed_model_v2(name="Lightweight_Proposed"):
    """
    Improved Lightweight CNN with:
    - Depthwise Separable Convolutions (parameter efficient)
    - Squeeze-and-Excitation channel attention
    - Residual skip connections
    - Progressive filter scaling: 32 → 64 → 128
    - Progressive dropout: 0.2 → 0.3 → 0.4 → 0.6 → 0.5

    Target: ~150-200K parameters, competitive with MobileNetV3-Small.

    Design rationale:
    - DSConv reduces params by ~8-9x vs standard Conv2D, allowing wider filters
    - SE blocks add <5% params but provide channel attention for disease-specific features
    - Residual connections improve gradient flow in this deeper architecture
    - Progressive dropout provides stronger regularization in deeper layers
    """
    inp = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3), name="input_layer")

    # Initial standard conv to expand channels from 3 → 32
    x = layers.Conv2D(
        32, 3, padding="same", kernel_regularizer=l2(config.L2_WEIGHT),
        name="stem_conv"
    )(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    # Block 1: 32 filters, 224→112
    x = _dwsconv_block(x, 32, "block1", dropout_rate=0.2, use_se=True, use_residual=True)

    # Block 2: 64 filters, 112→56
    x = _dwsconv_block(x, 64, "block2", dropout_rate=0.3, use_se=True, use_residual=True)

    # Block 3: 128 filters, 56→28
    x = _dwsconv_block(x, 128, "block3", dropout_rate=0.4, use_se=True, use_residual=True)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = layers.Dense(128, kernel_regularizer=l2(config.L2_WEIGHT), name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense1")(x)
    x = layers.Activation("relu", name="relu_dense1")(x)
    x = layers.Dropout(0.6, name="dropout_dense1")(x)

    x = layers.Dense(64, kernel_regularizer=l2(config.L2_WEIGHT), name="dense2")(x)
    x = layers.BatchNormalization(name="bn_dense2")(x)
    x = layers.Activation("relu", name="relu_dense2")(x)
    x = layers.Dropout(0.5, name="dropout_dense2")(x)

    out = layers.Dense(config.NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)

    model = Model(inputs=inp, outputs=out, name=name)
    return model


# ============================================================
# VGG16 (Transfer Learning)
# ============================================================
def create_vgg16_model():
    """
    VGG16 transfer learning model.
    138M total params, classic baseline used by Tariq et al. [25]
    and Theerthagiri et al. [26] on the same dataset.
    """
    from tensorflow.keras.applications import VGG16

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    )
    base_model.trainable = False

    inp = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = base_model(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=l2(config.L2_WEIGHT))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2(config.L2_WEIGHT))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = Model(inputs=inp, outputs=out, name="VGG16")
    return model, base_model


# ============================================================
# INCEPTIONV3 (Transfer Learning)
# ============================================================
def create_inceptionv3_model():
    """
    InceptionV3 transfer learning model.
    23.8M total params, used by Haque et al. [5] for maize disease detection.
    """
    from tensorflow.keras.applications import InceptionV3

    base_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    )
    base_model.trainable = False

    inp = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = base_model(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=l2(config.L2_WEIGHT))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2(config.L2_WEIGHT))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = Model(inputs=inp, outputs=out, name="InceptionV3")
    return model, base_model


# ============================================================
# DENSENET121 (Transfer Learning)
# ============================================================
def create_densenet121_model():
    """
    DenseNet121 transfer learning model.
    8M total params, popular in agricultural and medical imaging.
    """
    from tensorflow.keras.applications import DenseNet121

    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    )
    base_model.trainable = False

    inp = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = base_model(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2(config.L2_WEIGHT))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2(config.L2_WEIGHT))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = Model(inputs=inp, outputs=out, name="DenseNet121")
    return model, base_model


# ============================================================
# RESNET50 (Transfer Learning)
# ============================================================
def create_resnet50_model(unfreeze_layers=0):
    """
    Create ResNet50 transfer learning model.

    Args:
        unfreeze_layers: Number of layers to unfreeze from the top of base model.
    """
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    )

    # Freeze base model
    base_model.trainable = False

    if unfreeze_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

    inp = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = base_model(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=l2(config.L2_WEIGHT))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2(config.L2_WEIGHT))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = Model(inputs=inp, outputs=out, name="ResNet50")
    return model, base_model


# ============================================================
# MODEL COMPILATION
# ============================================================
def compile_model(model, learning_rate, loss_type=None, class_weights=None):
    """
    Compile a model with the specified loss function.

    Args:
        model: Keras model
        learning_rate: Learning rate for Adam optimizer
        loss_type: "categorical_crossentropy", "focal", or None (uses config)
        class_weights: Dict of class weights (used only for reference, actual weights
                       are passed during model.fit())
    """
    if loss_type is None:
        loss_type = config.LOSS_TYPE

    if loss_type == "focal":
        loss_fn = focal_loss(
            gamma=config.FOCAL_LOSS_GAMMA,
            alpha=config.FOCAL_LOSS_ALPHA,
        )
    else:
        loss_fn = "categorical_crossentropy"

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    return model


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal Loss for handling class imbalance."""
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1.0 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return focal_loss_fixed


# ============================================================
# MODEL INFO
# ============================================================
def print_model_info(model):
    """Print model information summary."""
    total_params = model.count_params()
    trainable = sum(K.count_params(w) for w in model.trainable_weights)
    non_trainable = total_params - trainable
    size_mb = total_params * 4 / (1024 * 1024)

    print(f"\n  Model: {model.name}")
    print(f"  Total parameters:       {total_params:>12,}")
    print(f"  Trainable parameters:   {trainable:>12,}")
    print(f"  Non-trainable params:   {non_trainable:>12,}")
    print(f"  Model size (approx):    {size_mb:>12.2f} MB")

    return {
        "total_params": total_params,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "size_mb": size_mb,
    }


# ============================================================
# ABLATION MODEL FACTORY
# ============================================================
def create_ablation_model(ablation_config, name=None):
    """Create a model variant for ablation study (V1 or V2)."""
    model_name = name or "Ablation_Model"

    if ablation_config.get("version") == "v2":
        # V2 ablation: create V2 with specific toggles
        return create_proposed_model_v2_ablation(
            use_se=ablation_config.get("use_se", True),
            use_residual=ablation_config.get("use_residual", True),
            name=model_name,
        )
    else:
        # V1 ablation: original architecture variants
        kwargs = {
            "filters": ablation_config.get("filters", [16, 32, 64]),
            "dropout_mode": ablation_config.get("dropout_mode", "progressive"),
            "dropout_rate": ablation_config.get("dropout_rate", 0.5),
            "use_bn": ablation_config.get("use_bn", True),
            "use_gap": ablation_config.get("use_gap", True),
            "use_se": ablation_config.get("use_se", False),
            "name": model_name,
        }
        return create_proposed_model(**kwargs)


def create_proposed_model_v2_ablation(use_se=True, use_residual=True, name="V2_Ablation"):
    """Create V2 model variant with toggleable SE and residual connections."""
    inp = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3), name="input_layer")

    x = layers.Conv2D(
        32, 3, padding="same", kernel_regularizer=l2(config.L2_WEIGHT),
        name="stem_conv"
    )(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    x = _dwsconv_block(x, 32, "block1", dropout_rate=0.2, use_se=use_se, use_residual=use_residual)
    x = _dwsconv_block(x, 64, "block2", dropout_rate=0.3, use_se=use_se, use_residual=use_residual)
    x = _dwsconv_block(x, 128, "block3", dropout_rate=0.4, use_se=use_se, use_residual=use_residual)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, kernel_regularizer=l2(config.L2_WEIGHT), name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense1")(x)
    x = layers.Activation("relu", name="relu_dense1")(x)
    x = layers.Dropout(0.6, name="dropout_dense1")(x)
    x = layers.Dense(64, kernel_regularizer=l2(config.L2_WEIGHT), name="dense2")(x)
    x = layers.BatchNormalization(name="bn_dense2")(x)
    x = layers.Activation("relu", name="relu_dense2")(x)
    x = layers.Dropout(0.5, name="dropout_dense2")(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)

    model = Model(inputs=inp, outputs=out, name=name)
    return model
