import cv2
import jax
import jax.numpy as jnp
import numpy as np


def random_crop(key, img, padding):
    crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((2,), dtype=jnp.int32)])
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop(key, imgs, padding=4):
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)



def random_cutout(key, img, max_size=16):
    """Apply random cutout to a single image.
    
    Args:
        key: JAX random key
        img: Input image array of shape (H, W, C)
        max_size: Maximum size of cutout square
    
    Returns:
        Image with random cutout applied
    """
    h, w = img.shape[:2]
    keys = jax.random.split(key, 2)
    
    # Sample cutout location
    start_x = jax.random.randint(keys[0], (), 0, w - max_size + 1)
    start_y = jax.random.randint(keys[1], (), 0, h - max_size + 1)
    
    # Create mask directly using dynamic_update_slice
    mask = jnp.ones_like(img)
    cutout = jnp.zeros((max_size, max_size) + img.shape[2:])
    
    # Place the cutout in the mask
    mask = jax.lax.dynamic_update_slice(
        mask, 
        cutout,
        (start_y, start_x) + (0,) * len(img.shape[2:])
    )
    # cv2.imwrite("sample.jpg",(np.array(img*mask)*255).astype(np.uint8))
    return img * mask

def batched_random_cutout(key, imgs, max_size=16):
    """Apply random cutout to a batch of images.
    
    Args:
        key: JAX random key
        imgs: Batch of images of shape (B, H, W, C)
        max_size: Maximum size of cutout square
    
    Returns:
        Batch of images with random cutouts applied
    """
    return jax.vmap(lambda k, x: random_cutout(k, x, max_size))(
        jax.random.split(key, imgs.shape[0]), 
        imgs
    )




# def adjust_brightness(image: jnp.Array, delta:float):
#   return image + jnp.asarray(delta, image.dtype)

# def random_brightness(
#     key: jax.PRNGKey,
#     image: jnp.Array,
#     max_delta: float=0.1,
# ) -> jnp.Array:
#   """`adjust_brightness(...)` with random delta in `[-max_delta, max_delta)`."""
#   # DO NOT REMOVE - Logging usage.
#   delta = jax.random.uniform(key, (), minval=-max_delta, maxval=max_delta)
#   return adjust_brightness(image, delta)

# def batched_random_cutout(key, imgs):
#     keys = jax.random.split(key, imgs.shape[0])
#     rng = jax.random.uniform(key)
   
#     return jax.lax.cond(
#         rng < 0.1,
#         lambda x: jax.vmap(random_cutout, (0,0))(keys, imgs),
#         lambda x: x,
#         imgs)

