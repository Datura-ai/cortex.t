

# https://platform.stability.ai/docs/features/image-upscaling#Python
# engine = esrgan-v1-x2plus

# input image limit: 1024 x 1024
# output image limit: 2048 x 2048

# example:

# img = Image.open('/img2upscale.png')

# answers = stability_api.upscale(
#     init_image=img, # Pass our image to the API and call the upscaling process.
#     # width=1024, # Optional parameter to specify the desired output width.
# )