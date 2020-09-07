# Tutorial link: https://medium.com/mlreview/making-ai-art-with-style-transfer-using-keras-8bb5fa44b216


import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from scipy.optimize import fmin_l_bfgs_b
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
tf.disable_v2_behavior()

# Set up image paths: original, style, and output

image_path = '/Users/shreyasravi/Desktop/Python/AIDraw/images/image.jpeg'
style_path = '/Users/shreyasravi/Desktop/Python/AIDraw/images/style.jpeg'
output_path = '/Users/shreyasravi/Desktop/Python/AIDraw/output/output.jpeg'

# Width and height of image

height = 512
width = 512
size = (height, width)

# Load images into arrays

image_orig = Image.open(image_path)
image_orig_size = image_orig.size
image = load_img(path=image_path, target_size=size)
image_arr = img_to_array(image)
image_arr = K.variable(preprocess_input(np.expand_dims(image_arr, axis=0)), dtype='float32')

style = load_img(path=style_path, target_size=size)
style_arr = img_to_array(style)
style_arr = K.variable(preprocess_input(np.expand_dims(style_arr, axis=0)), dtype='float32')

output = np.random.randint(256, size=(width, height, 3)).astype('float64')
output = preprocess_input(np.expand_dims(output, axis=0))

output_placeholder = K.placeholder(shape=(1, width, height, 3))


# Content Loss Function: L_c(p, x, l) = 1/2 * ∑(F^l - P^l)^2
# F & P are matrices M x N
# N is the number of filters in layer l and M is the number of spatial elements in the feature map for layer l
# F contains the feature representation of x for layer l, P contains the feature representation of p for layer l

def get_feature_reps(x, layer_names, model):
    # Get feature representations of input x for one or more layers in a given model

    feat_matrices = []
    for ln in layer_names:
        selected_layer = model.get_layer(ln)
        feat_raw = selected_layer.output
        feat_raw_shape = K.shape(feat_raw).eval(session=tf_session)
        n_l = feat_raw_shape[-1]
        m_l = feat_raw_shape[1] * feat_raw_shape[2]
        feat_matrix = K.reshape(feat_raw, (m_l, n_l))
        feat_matrix = K.transpose(feat_matrix)
        feat_matrices.append(feat_matrix)
    return feat_matrices


def get_content_loss(F, P):
    c_loss = 0.5 * K.sum(K.square(F - P))
    return c_loss

# Gram matrix is a square matrix that contains the dot product between each vectorized filter in layer l

def get_gram_matrix(f):
    g = K.dot(f, K.transpose(f))
    return g

# Style loss for a given layer: E_l = 1/(4 M^2 N^2) * ∑(G^l - A^l)^2
# G is the gram matrix for the output image x, and A is the gram matrix for the style image a

# Style loss function = L_s(a, x, l) = ∑w_l * E_l
# w_l is the style loss weight for each layer
def get_style_loss(ws, Gs, As):
    s_loss = K.variable(0.)
    for w, G, A in zip(ws, Gs, As):
        m_l = K.int_shape(G)[1]
        n_l = K.int_shape(G)[0]
        g_gram = get_gram_matrix(G)
        a_gram = get_gram_matrix(A)
        loss = w * 0.25 * K.sum(K.square(g_gram - a_gram)) / (n_l ** 2 * m_l ** 2)
        s_loss = s_loss + loss
        # s_loss = s_loss.assign_add(loss)
    return s_loss

# Total loss function: L(p, a, x, l) = alpha * L_c(p, x, l) + beta * L_s(a, x, l)

def get_total_loss(output_placeholder, alpha=1.0, beta=10000.0):
    f = get_feature_reps(output_placeholder, layer_names=[image_layer_name], model=output_model)[0]
    gs = get_feature_reps(output_placeholder, layer_names=style_layer_names, model=output_model)
    content_loss = get_content_loss(f, P)
    style_loss = get_style_loss(ws, gs, As)
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss

# Calculate the total loss using k.function

def calculate_loss(output_arr):
    """
    Calculate total loss using K.function
    """
    if output_arr.shape != (1, width, width, 3):
        output_arr = output_arr.reshape((1, width, height, 3))
    loss_fcn = K.function([output_model.input], [get_total_loss(output_model.input)])
    return loss_fcn([output_arr])[0].astype('float64')

#  Calculate the gradient of the loss function with respect to the generated image

def get_grad(output_arr):
    """
    Calculate the gradient of the loss function with respect to the generated image
    """
    if output_arr.shape != (1, width, height, 3):
        output_arr = output_arr.reshape((1, width, height, 3))
    grad_fcn = K.function([output_model.input], K.gradients(get_total_loss(output_model.input), [output_model.input]))
    grad = grad_fcn([output_arr])[0].flatten().astype('float64')
    return grad


def post_process_array(x):
    # Zero-center by mean pixel
    if x.shape != (width, height, 3):
        x = x.reshape((width, height, 3))
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x


def reprocess_array(x):
    x = np.expand_dims(x.astype('float64'), axis=0)
    x = preprocess_input(x)
    return x


def save_original_size(x, target_size=image_orig_size):
    x_im = Image.fromarray(x)
    x_im = x_im.resize(target_size)
    x_im.save(output_path)
    return x_im

tf_session = tf.compat.v1.keras.backend.get_session()

image_model = VGG16(include_top=False, weights='imagenet', input_tensor=image_arr)
style_model = VGG16(include_top=False, weights='imagenet', input_tensor=style_arr)
output_model = VGG16(include_top=False, weights='imagenet', input_tensor=output_placeholder)
image_layer_name = 'block4_conv2'
style_layer_names = [
                    'block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    # 'block5_conv1'
                    ]

P = get_feature_reps(x=image_arr, layer_names=[image_layer_name], model=image_model)[0]
As = get_feature_reps(x=style_arr, layer_names=style_layer_names, model=style_model)
ws = np.ones(len(style_layer_names)) / float(len(style_layer_names))

iterations = 100
x_val = output.flatten()
start = time.time()
x_opt, f_val, info = fmin_l_bfgs_b(calculate_loss, x_val, fprime=get_grad,
                                   maxiter=iterations, disp=True)
xOut = post_process_array(x_opt)
xIm = save_original_size(xOut)
print('Image saved')
end = time.time()
print('Time taken: {}'.format(end - start))
